"""
Self-contained training script for lvl1_vqgan configuration.
Imports data utilities from prepare.py; everything else is inlined here.
Run: python train.py
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import os
import threading
import warnings

import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Callable, Iterable, Literal, Protocol

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    Timer,
)
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning_utilities.core.rank_zero import rank_zero_info
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from prepare import (
    LearningParameters,
    MelSpecParameters,
    SimpleMelSpecConverter,
    COMPOSITE_WEIGHTS,
    build_data_module,
    cdpam_distance,
    chroma_cosine_distance,
    make_cdpam_evaluator,
    multi_res_stft_distance,
    phase_distance,
)

# ===========================================================================
# Transform functions
# ===========================================================================


def transparent(x: torch.Tensor) -> torch.Tensor:
    return x


def tanh_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def log_normal(x: torch.Tensor) -> torch.Tensor:
    log_transformed = torch.log(x + 1e-6)
    mean = torch.mean(log_transformed)
    std = torch.std(log_transformed)
    return (log_transformed - mean) / std


# ===========================================================================
# Waveform tokenization utility
# ===========================================================================


def quantize_waveform_256(waveform: torch.Tensor) -> torch.Tensor:
    return ((waveform + 1) * 127.5).ceil().to(torch.uint8)


# ===========================================================================
# EMA callback and optimizer
# ===========================================================================


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(ema_model_tuple, current_model_tuple, alpha=(1.0 - decay))


def run_ema_update_cpu(
    ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None
):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()
    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
    ):
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.save_original_optimizer_state = False
        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None
        self.ema_params = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> Iterable[torch.Tensor]:
        return (param for group in self.param_groups for param in group["params"])

    def step(self, closure=None, **kwargs):
        self.join()
        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()
            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())
            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device)
                for param in opt_params[len(self.ema_params) :]
            )
            self.rebuild_ema_params = False

        loss = self.optimizer.step(closure)
        if self._should_update_at_step():
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())  # type: ignore
        with torch.cuda.stream(self.stream):  # type: ignore
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True)
                for param in self.all_parameters()
            )
            if self.device.type == "cuda":
                ema_update(self.ema_params, current_model_state, self.decay)
        if self.device.type == "cpu":
            self.thread = threading.Thread(
                target=run_ema_update_cpu,
                args=(self.ema_params, current_model_state, self.decay, self.stream),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False):
        self.join()
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def join(self):
        if self.stream is not None:
            self.stream.synchronize()
        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        self.join()
        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()
        ema_params = (
            self.ema_params
            if not self.in_saving_ema_model_context
            else list(self.all_parameters())
        )
        return {
            "opt": self.optimizer.state_dict(),
            "ema": ema_params,
            "current_step": self.current_step,
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }

    def load_state_dict(self, state_dict):
        self.join()
        self.optimizer.load_state_dict(state_dict["opt"])
        self.ema_params = tuple(
            param.to(self.device) for param in copy.deepcopy(state_dict["ema"])
        )
        self.current_step = state_dict["current_step"]
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True


class EMA(L.Callback):
    def __init__(
        self,
        decay: float,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
    ):
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        device = pl_module.device if not self.cpu_offload else torch.device("cpu")
        trainer.optimizers = [
            EMAOptimizer(
                optim,
                device=device,
                decay=self.decay,
                every_n_steps=self.every_n_steps,
                current_step=trainer.global_step,
            )
            for optim in trainer.optimizers
            if not isinstance(optim, EMAOptimizer)
        ]

    def on_validation_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: L.Trainer) -> bool:
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: L.Trainer) -> bool:
        return any(isinstance(opt, EMAOptimizer) for opt in trainer.optimizers)

    def swap_model_weights(self, trainer: L.Trainer, saving_ema_model: bool = False):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.switch_main_parameter_weights(saving_ema_model)

    @contextlib.contextmanager
    def save_ema_model(self, trainer: L.Trainer):
        self.swap_model_weights(trainer, saving_ema_model=True)
        try:
            yield
        finally:
            self.swap_model_weights(trainer, saving_ema_model=False)

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: L.Trainer):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.save_original_optimizer_state = True
        try:
            yield
        finally:
            for optimizer in trainer.optimizers:
                optimizer.save_original_optimizer_state = False  # type: ignore

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_callback = trainer.checkpoint_callback
        connector = trainer._checkpoint_connector
        ckpt_path = connector.resume_checkpoint_path  # type: ignore
        if (
            ckpt_path
            and checkpoint_callback is not None
            and "NeMo" in type(checkpoint_callback).__name__
        ):
            ext = checkpoint_callback.FILE_EXTENSION  # type: ignore
            if ckpt_path.endswith(f"-EMA{ext}"):
                rank_zero_info(
                    "loading EMA based weights. The callback will treat the loaded EMA weights "
                    "as the main weights and create a new EMA copy when training."
                )
                return
            ema_path = ckpt_path.replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))
                checkpoint["optimizer_states"] = ema_state_dict["optimizer_states"]
                del ema_state_dict
                rank_zero_info("EMA state has been restored.")
            else:
                raise MisconfigurationException(
                    f"Unable to find associated EMA weights at: {ema_path}"
                )


# ===========================================================================
# Trainer factory
# ===========================================================================


def get_trainer(
    learning_parameters: LearningParameters, minutes: int = 9 * 60
) -> L.Trainer:
    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available, using CPU")
        devices = "auto"
        accelerator = "cpu"
    else:
        devices = learning_parameters.devices
        accelerator = (
            "cpu" if learning_parameters.devices == "cpu" or devices == "cpu" else "gpu"
        )

    save_folder = learning_parameters.save_path
    ema = EMA(learning_parameters.beta_ema)
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(
        save_dir=save_folder, name=learning_parameters.model_name
    )
    loggers: list[Logger] = [tensorboard_logger]

    best_checkpoint_callback = ModelCheckpoint(
        dirpath=save_folder,
        filename=learning_parameters.model_name + "_best",
        save_weights_only=False,
        save_top_k=1,
        monitor=learning_parameters.loss_monitor,
        enable_version_counter=False,
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=save_folder,
        filename=learning_parameters.model_name + "_last",
        save_weights_only=False,
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
    )
    early_stopping = EarlyStopping(
        monitor=learning_parameters.loss_monitor,
        stopping_threshold=learning_parameters.trigger_loss,
        patience=int(learning_parameters.epochs),
    )

    precision = "bf16-mixed" if learning_parameters.amp else 32
    model_summary = ModelSummary(max_depth=2)
    timer = Timer(duration={"minutes": minutes})

    trainer = L.Trainer(
        gradient_clip_val=learning_parameters.gradient_clip,
        logger=loggers,
        callbacks=[
            early_stopping,
            best_checkpoint_callback,
            last_checkpoint_callback,
            model_summary,
            learning_rate_monitor,
            ema,
            timer,
        ],
        strategy="ddp_find_unused_parameters_true",
        devices=devices,
        max_epochs=learning_parameters.epochs,
        log_every_n_steps=1,
        precision=precision,
        accelerator=accelerator,
        accumulate_grad_batches=learning_parameters.grad_accumulation,
        limit_train_batches=learning_parameters.limit_train_batches,
        limit_val_batches=learning_parameters.limit_eval_batches,
        limit_test_batches=learning_parameters.limit_test_batches,
    )
    return trainer


# ===========================================================================
# Loss components
# ===========================================================================


@dataclass
class LossOutput:
    total: torch.Tensor
    individual: dict[str, torch.Tensor]


class LossComponentBase(ABC):
    name: str
    differentiable: bool
    weight: float

    @abstractmethod
    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor: ...


class LossAggregatorProto(Protocol):
    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput: ...

    def recalculate_total_loss(self, loss: LossOutput) -> LossOutput: ...


@dataclass
class RecLoss:
    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str
    ref_key: str
    differentiable: bool = True
    transform_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    phase_parameter: int = 1

    def __call__(self, estimation, target):
        pred_slice = estimation[self.pred_key]
        target_slice = target[self.ref_key]
        return self._phased_loss(
            self.transform_func(pred_slice),
            self.transform_func(target_slice),
            phase_parameter=self.phase_parameter,
        )

    def _phased_loss(self, estimation, target, phase_parameter: int = 10):
        loss_vector = torch.zeros(phase_parameter * 2).to(estimation.device)
        for idx in range(phase_parameter):
            if idx == 0:
                loss_vector[idx * 2] = self.base_loss(estimation, target)
                loss_vector[idx * 2 + 1] = loss_vector[idx * 2] + 1e-6
                continue
            loss_vector[idx * 2] = self.base_loss(
                estimation[:, :, idx:], target[:, :, :-idx]
            )
            loss_vector[idx * 2 + 1] = self.base_loss(
                estimation[:, :, :-idx], target[:, :, idx:]
            )
        return loss_vector.min()


@dataclass
class EdgeRecLoss:
    name: str
    weight: float
    base_loss: nn.Module
    pred_key: str
    ref_key: str
    differentiable: bool = True
    transform_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    edge_power: float = 1.0

    def __call__(self, estimation, target):
        pred_slice = estimation[self.pred_key]
        batch_size, channels, length = pred_slice.shape
        target_slice = target[self.ref_key]
        linspace_slice = torch.arange(0, length).to(pred_slice.device) / (length - 1)
        weight_slice = (
            4 * (1 - linspace_slice * (1 - linspace_slice)) ** self.edge_power
        )
        weight_slice = weight_slice.repeat(batch_size, channels, 1)
        return self.base_loss(pred_slice * weight_slice, target_slice * weight_slice)


@dataclass
class AlignLoss:
    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True

    def __call__(self, estimation, target):
        emb = estimation["emb"]
        z_e = target["z_e"]
        loss = torch.tensor(0.0).to(z_e.device)
        for codebook_idx in range(emb.shape[1]):
            loss = loss + self.base_loss(emb[:, codebook_idx, ...], z_e.detach())
        return loss


@dataclass
class CommitLoss:
    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True

    def __call__(self, estimation, target):
        emb = estimation["emb"]
        z_e = target["z_e"]
        loss = torch.tensor(0.0).to(z_e.device)
        for codebook_idx in range(emb.shape[1]):
            loss = loss + self.base_loss(emb[:, codebook_idx, ...].detach(), z_e)
        return loss


@dataclass
class MelSpecLoss:
    name: str
    weight: float
    pred_key: str
    ref_key: str
    base_loss: nn.Module
    mel_spec_converter: SimpleMelSpecConverter
    transform_func: Callable[[torch.Tensor], torch.Tensor] = (
        lambda x: torch.tanh(x) * 2 - 1
    )
    lin_start: float = 1.0
    lin_end: float = 1.0
    differentiable: bool = True

    def _mel_spec_and_process(self, x: torch.Tensor) -> torch.Tensor:
        lin_vector = torch.linspace(
            self.lin_start, self.lin_end, self.mel_spec_converter.mel_spec.n_mels
        )
        eye_mat = torch.diag(lin_vector).to(x.device)
        mel_out = self.mel_spec_converter.convert(x.flatten(start_dim=0, end_dim=1))
        mel_out = self.transform_func(eye_mat @ mel_out)
        return mel_out

    def __call__(self, estimation, target):
        pred_slice = estimation[self.pred_key]
        target_slice = target[self.ref_key]
        self.mel_spec_converter.mel_spec = self.mel_spec_converter.mel_spec.to(
            pred_slice.device
        )
        return self.base_loss(
            self._mel_spec_and_process(pred_slice),
            self._mel_spec_and_process(target_slice),
        )


@dataclass
class CDPAMLoss:
    """Directly optimize the CDPAM perceptual metric (the eval target).

    CDPAM.forward() runs under torch.no_grad(), so we replicate its computation
    (base_encoder embeddings -> L2 normalize -> dist_model) WITHOUT no_grad to
    let gradients flow back to the generator. The pretrained CDPAM weights are
    frozen. Inputs are resampled to 22050 Hz and scaled to int16 range, matching
    prepare.cdpam_distance.
    """

    name: str
    weight: float
    pred_key: str
    ref_key: str
    src_sr: int = 44100
    cdpam_sr: int = 22050
    differentiable: bool = True
    _evaluator: Any = None
    _resampler: Any = None

    def _ensure(self, device: torch.device) -> None:
        if self._evaluator is None:
            import cdpam
            import torchaudio

            self._evaluator = cdpam.CDPAM(dev=str(device))
            for p in self._evaluator.model.parameters():
                p.requires_grad_(False)
            self._resampler = torchaudio.transforms.Resample(
                self.src_sr, self.cdpam_sr
            ).to(device)

    def __call__(self, estimation, target):
        pred = estimation[self.pred_key]
        ref = target[self.ref_key]
        self._ensure(pred.device)
        # Replicate CDPAM.forward (which uses no_grad): base_encoder -> acoustics
        # embedding (2nd of 3 returns) -> L2 normalize -> model_dist. Inputs are
        # [N, L] at 22050 Hz scaled to int16 range; encoder does unsqueeze(1).
        pred_w = (self._resampler(pred.float()) * 32768.0).reshape(pred.shape[0], -1)
        ref_w = (self._resampler(ref.float()) * 32768.0).reshape(ref.shape[0], -1)
        model = self._evaluator.model
        # bf16 autocast for the frozen cdpam net: it is Conv1d/BN/relu (no complex
        # ops), so half precision is safe and ~halves this loss's compute/memory.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, a1, _ = model.base_encoder.forward(ref_w.unsqueeze(1))
            _, a2, _ = model.base_encoder.forward(pred_w.unsqueeze(1))
            a1 = F.normalize(a1, dim=1)
            a2 = F.normalize(a2, dim=1)
            dist = model.model_dist.forward(a1, a2).mean()
        return dist.float()


@dataclass
class MultiResSTFTLoss:
    """Multi-resolution STFT loss (spectral convergence + log-mag L1).

    Overall harmonic + transient fidelity across FFT sizes; the standard
    neural-vocoder reconstruction loss. Differentiable (torch.stft).
    """

    name: str
    weight: float
    pred_key: str = "slice"
    ref_key: str = "slice"
    ffts: tuple[int, ...] = (512, 1024, 2048)
    differentiable: bool = True

    def __call__(self, estimation, target):
        return multi_res_stft_distance(
            estimation[self.pred_key], target[self.ref_key], ffts=self.ffts
        )


@dataclass
class ChromaLoss:
    """Chroma (pitch-class) cosine distance — the MELODY/harmony term.

    Maps STFT magnitude to 12 pitch classes via a fixed librosa filterbank and
    penalizes per-frame cosine mismatch. Robust to timbre/octave; targets the
    melodic content cdpam is blind to. Differentiable.
    """

    name: str
    weight: float
    pred_key: str = "slice"
    ref_key: str = "slice"
    sr: int = 44100
    n_fft: int = 2048
    differentiable: bool = True

    def __call__(self, estimation, target):
        return chroma_cosine_distance(
            estimation[self.pred_key],
            target[self.ref_key],
            sr=self.sr,
            n_fft=self.n_fft,
        )


@dataclass
class PhaseLoss:
    """Magnitude-weighted multi-resolution phase incoherence.

    Penalizes phase mismatch (cos of phase diff, weighted by target energy)
    across FFT sizes — the phase coherence that pitched/melodic content needs and
    that the magnitude-only mrstft/chroma terms are blind to. Differentiable.
    """

    name: str
    weight: float
    pred_key: str = "slice"
    ref_key: str = "slice"
    ffts: tuple[int, ...] = (512, 1024, 2048)
    differentiable: bool = True

    def __call__(self, estimation, target):
        return phase_distance(
            estimation[self.pred_key], target[self.ref_key], ffts=self.ffts
        )


@dataclass
class DiscriminatorHingeLoss:
    name: str
    weight: float
    pred_key_real: str
    pred_key_fake: str
    differentiable: bool = True

    def __call__(self, pred, _):
        loss_real = torch.mean(F.relu(1 - pred[self.pred_key_real][..., 0]))
        loss_fake = torch.mean(F.relu(1 + pred[self.pred_key_fake][..., 0]))
        return 0.5 * (loss_real + loss_fake)


@dataclass
class GeneratorHingeLoss:
    name: str
    weight: float
    pred_key_disc: str
    differentiable: bool = True

    def __call__(self, pred, _):
        return torch.mean(torch.clamp(1 - pred[self.pred_key_disc][..., 0], min=0))


class WeightedSumAggregator:
    def __init__(self, components: Iterable) -> None:
        self.components = components

    def __call__(self, pred, target) -> LossOutput:
        loss = LossOutput(torch.tensor(0.0), {})
        for component in self.components:
            ind_loss = component(pred, target)
            if component.differentiable:
                loss.total = loss.total.to(ind_loss.device)
                loss.total = loss.total + component.weight * ind_loss
            loss.individual[component.name] = ind_loss
        return loss

    def recalculate_total_loss(self, loss: LossOutput) -> LossOutput:
        recalculated = LossOutput(torch.tensor(0.0).to(loss.total.device), {})
        for component in self.components:
            if component.differentiable:
                recalculated.total = recalculated.total + (
                    component.weight * loss.individual[component.name]
                )
            recalculated.individual[component.name] = loss.individual[component.name]
        return recalculated


# ===========================================================================
# Codebook implementation
# ===========================================================================


class VQCodeBookFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in: torch.Tensor, embedding_weights: torch.Tensor):
        ctx.batch_size = x_in.shape[0]
        embedding_batch = embedding_weights.unsqueeze(0).repeat((x_in.shape[0], 1, 1))
        x_in_t = x_in.transpose(1, 2).contiguous().float()
        embedding_batch_t = embedding_batch.transpose(1, 2).contiguous().float()
        embedding_batch_flat = embedding_batch_t.flatten(start_dim=0, end_dim=1)
        distances = torch.cdist(x_in_t, embedding_batch_t)
        indices = torch.argmin(distances, dim=2, keepdim=True)
        x_out = torch.index_select(embedding_batch_flat, dim=0, index=indices.flatten())
        x_out = x_out.contiguous().view((x_in.shape[0], x_in.shape[2], x_in.shape[1]))
        x_out = x_out.transpose(1, 2).contiguous()
        ctx.save_for_backward(embedding_weights, indices)
        return x_out, indices

    @staticmethod
    def backward(ctx, grad_outputs, indices):
        grad_input = None
        grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_outputs
        if ctx.needs_input_grad[1]:
            embedding_weights, indices = ctx.saved_tensors
            grad_emb = torch.zeros_like(embedding_weights)
            for batch_idx, batch in enumerate(indices.flatten(start_dim=1)):
                running_idx = 0
                for idx in batch:
                    idx_value = idx.item()
                    grad_emb[:, idx_value] = (
                        grad_emb[:, idx_value]
                        + grad_outputs[batch_idx, :, running_idx]
                        / indices.flatten().shape[0]
                    )  # type: ignore
                    running_idx += 1
        return grad_input, grad_emb, None, None


def vq_code_book_select(x_in: torch.Tensor, emb_batch: torch.Tensor):
    return VQCodeBookFunc.apply(x_in, emb_batch)


class CodeBook(ABC, nn.Module):
    @abstractmethod
    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def apply_codebook(self, x_in: torch.Tensor, code_sg: bool = False) -> tuple: ...

    @abstractmethod
    def update_usage(self, min_enc: torch.Tensor) -> None: ...

    @abstractmethod
    def reset_usage(self) -> None: ...

    @abstractmethod
    def random_restart(self) -> float: ...


class VQCodeBook(CodeBook):
    def __init__(
        self, token_dim: int, num_tokens: int = 512, usage_threshold: float = 1e-9
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        # randn (full space, not positive orthant) scaled to ~unit-norm codes
        self.code_embedding = nn.Parameter(
            torch.randn(num_tokens, token_dim) / token_dim**0.5
        )
        self.usage_threshold = usage_threshold
        self.register_buffer("usage", torch.ones(self.num_tokens), persistent=False)

    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        return self.code_embedding[indices]

    def apply_codebook(self, x_in: torch.Tensor, code_sg: bool = False):
        # Euclidean VQ on raw vectors: codes keep their magnitude so residual
        # quantization (RQ) actually refines — normalizing here forced every RQ
        # step to emit a unit-norm code for an ever-smaller residual.
        embedding_weights = self.code_embedding.transpose(0, 1).contiguous()
        z_q, indices = vq_code_book_select(
            x_in, embedding_weights.detach() if code_sg else embedding_weights
        )  # type: ignore
        self.update_usage(indices)
        return z_q.unsqueeze(1), indices

    def update_usage(self, min_enc: torch.Tensor) -> None:
        self.usage[min_enc.flatten()] = self.usage[min_enc.flatten()] + 1  # type: ignore
        self.usage /= 2  # type: ignore

    def reset_usage(self) -> None:
        self.usage.zero_()  # type: ignore

    def random_restart(self) -> float:
        dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)  # type: ignore
        alive_codes = torch.nonzero(self.usage >= self.usage_threshold).squeeze(1)  # type: ignore
        if len(dead_codes) > 0 and len(alive_codes) > 0:
            # revive dead codes near ALIVE ones (splits live clusters); copying
            # from arbitrary codes could just clone another dead code
            src = alive_codes[torch.randint(len(alive_codes), (len(dead_codes),))]
            with torch.no_grad():
                jitter = 0.1 * self.code_embedding[src].std()
                self.code_embedding[dead_codes] = self.code_embedding[
                    src
                ] + jitter * torch.randn_like(self.code_embedding[src])
        return dead_codes.shape[0]


class RQCodeBook(VQCodeBook):
    def __init__(
        self,
        token_dim: int,
        num_rq_steps: int,
        num_tokens: int = 512,
        usage_threshold: float = 1e-9,
    ) -> None:
        super().__init__(token_dim, num_tokens, usage_threshold)
        self._num_rq_steps = num_rq_steps

    def apply_codebook(self, x_in: torch.Tensor, code_sg: bool = False):
        x_res = x_in.clone()
        z_q_aggregated = []
        indices = []
        for _ in range(self._num_rq_steps):
            z_q_ind, indices_ind = super().apply_codebook(x_res, code_sg)
            x_res = x_res - z_q_ind.squeeze(1)
            z_q_aggregated.append(
                z_q_ind if len(z_q_aggregated) == 0 else z_q_aggregated[-1] + z_q_ind
            )
            indices.append(indices_ind)
        return torch.cat(z_q_aggregated, dim=1), torch.cat(indices, dim=-1)

    def embed_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        emb = torch.zeros(
            indices.size(0),
            indices.size(1),
            self.code_embedding.size(1),
            device=indices.device,
        )
        for idx in range(self._num_rq_steps):
            emb = emb + self.code_embedding[indices[..., idx]]
        return emb


class VQ1D(nn.Module):
    def __init__(
        self, token_dim: int, num_tokens: int = 8192, num_rq_steps: int = 1
    ) -> None:
        super().__init__()
        self.vq_codebook = RQCodeBook(
            token_dim, num_rq_steps=num_rq_steps, num_tokens=num_tokens
        )

    def forward(
        self, z_e: torch.Tensor, extract_losses: bool = False
    ) -> dict[str, torch.Tensor]:
        z_q, indices = self.vq_codebook.apply_codebook(z_e, code_sg=True)
        output = {"indices": indices, "v_q": z_q}
        if extract_losses:
            emb, _ = self.vq_codebook.apply_codebook(z_e.detach())
            output.update({"emb": emb})
        return output


# ===========================================================================
# Residual blocks
# ===========================================================================


def _sep_conv1d(
    num_channels: int, kernel_size: int, dilation: int, padding: int
) -> nn.Sequential:
    """Depthwise-separable conv: depthwise k-conv + pointwise 1x1 (~5x cheaper
    than a full conv at hidden=512, k=5 -> more optimizer steps per budget)."""
    return nn.Sequential(
        nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=num_channels,
        ),
        nn.Conv1d(num_channels, num_channels, kernel_size=1),
    )


class Res1DBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        dilation_factor: int,
        kernel_size: int,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        self.activation = activation_fn
        self.res_block_modules = nn.ModuleList([])
        for idx in range(num_res_conv):
            dilation = dilation_factor**idx
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
            if idx != num_res_conv - 1:
                self.res_block_modules.append(
                    nn.Sequential(
                        _sep_conv1d(num_channels, kernel_size, dilation, padding),
                        self.activation,
                        nn.BatchNorm1d(num_channels),
                    )
                )
            else:
                self.res_block_modules.append(
                    nn.Sequential(
                        _sep_conv1d(num_channels, kernel_size, dilation, padding),
                        self.activation,
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_init = x.clone()
        for seq_module in self.res_block_modules:
            x = seq_module(x)
        return x + x_init


class Res1DBlockReverse(Res1DBlock):
    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        dilation_factor: int,
        kernel_size: int,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__(
            num_channels, num_res_conv, dilation_factor, kernel_size, activation_fn
        )
        self.res_block_modules = nn.ModuleList([])
        for idx in range(num_res_conv):
            dilation = dilation_factor ** (num_res_conv - idx - 1)
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
            if idx != num_res_conv - 1:
                self.res_block_modules.append(
                    nn.Sequential(
                        _sep_conv1d(num_channels, kernel_size, dilation, padding),
                        self.activation,
                        nn.BatchNorm1d(num_channels),
                    )
                )
            else:
                self.res_block_modules.append(
                    nn.Sequential(
                        _sep_conv1d(num_channels, kernel_size, dilation, padding),
                        self.activation,
                    )
                )


# ===========================================================================
# Encoder
# ===========================================================================


# ===========================================================================
# Decoder
# ===========================================================================


class ISTFT(nn.Module):
    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )  # type: ignore
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]  # type: ignore
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)  # type: ignore
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope
        return y


class DecoderBase(ABC, nn.Module):
    @property
    @abstractmethod
    def last_layer(self) -> nn.Module: ...


# ===========================================================================
# Temporal codec (EnCodec-style layout): STFT freq bins -> CHANNELS, conv
# stride along TIME only. Latent = one vector per ~17ms (T_lat frames/slice).
# Fixes the legacy grid layout where 0.74s collapsed to 4-8 time positions
# and freq was interleaved into the sequence — which made the decoder emit a
# time-averaged spectrum (noise wash) regardless of latent precision.
# ===========================================================================


class AttnBlock1D(nn.Module):
    """Pre-norm transformer block over the time axis of a (B, C, T) latent.
    Convs are local; this adds global temporal context at the bottleneck."""

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self._norm1 = nn.LayerNorm(dim)
        self._attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self._norm2 = nn.LayerNorm(dim)
        self._mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)  # (B, T, C)
        n = self._norm1(h)
        h = h + self._attn(n, n, n, need_weights=False)[0]
        h = h + self._mlp(self._norm2(h))
        return h.transpose(1, 2)


class GRUBlock1D(nn.Module):
    """Pre-norm bidirectional GRU over the time axis of a (B, C, T) latent.
    Recurrent alternative to attention for global temporal context."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self._norm = nn.LayerNorm(dim)
        self._gru = nn.GRU(
            dim, dim // 2, batch_first=True, bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)  # (B, T, C)
        h = h + self._gru(self._norm(h))[0]
        return h.transpose(1, 2)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int = 512,
        hidden: int = 512,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        num_res_blocks: int = 2,
        num_res_conv: int = 3,
        kernel_size: int = 5,
        dilation_factor: int = 2,
        time_downsample: int = 2,
    ) -> None:
        super().__init__()
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        in_dim = (n_fft // 2 + 1) * 3  # real+imag+magnitude stacked as channels
        self._proj_in = nn.Conv1d(in_dim, hidden, kernel_size=7, padding=3)
        self._res1 = nn.Sequential(
            *[
                Res1DBlock(hidden, num_res_conv, dilation_factor, kernel_size)
                for _ in range(num_res_blocks)
            ]
        )
        self._down: nn.Module = (
            nn.Identity()
            if time_downsample == 1
            else nn.Conv1d(
                hidden,
                hidden,
                kernel_size=2 * time_downsample,
                stride=time_downsample,
                padding=time_downsample // 2,
            )
        )
        self._res2 = nn.Sequential(
            *[
                Res1DBlock(hidden, num_res_conv, dilation_factor, kernel_size)
                for _ in range(num_res_blocks)
            ]
        )
        self._attn = GRUBlock1D(hidden)
        self._proj_out = nn.Conv1d(hidden, token_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) waveform
        window = torch.hann_window(self._win_length).to(x.device)
        spec = torch.stft(
            x.flatten(start_dim=1),
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            window=window,
            return_complex=True,
        )  # (B, n_fft/2+1, T+1)
        n_frames = x.shape[-1] // self._hop_length
        feats = torch.cat([spec.real, spec.imag, spec.abs()], dim=1)[..., :n_frames]
        h = self._proj_in(feats)
        h = self._res1(h)
        h = self._down(h)
        h = self._res2(h)
        h = self._attn(h)
        z_e = self._proj_out(h)  # (B, token_dim, n_frames / time_downsample)
        # Pin z_e scale so it can't run away: unconstrained conv output inflates
        # under high LR -> alignment/commitment MSE (both vs z_e) grow unbounded.
        # l2 normalize ONCE here (unit-norm per token, matches codebook init scale
        # ~1) -- NOT inside the RQ select loop, which would force every residual
        # step to unit-norm and kill refinement.
        return F.normalize(z_e, dim=1)


class TemporalDecoder(DecoderBase):
    def __init__(
        self,
        token_dim: int = 512,
        hidden: int = 512,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        num_res_blocks: int = 2,
        num_res_conv: int = 3,
        kernel_size: int = 5,
        dilation_factor: int = 2,
        time_upsample: int = 2,
        output_conv_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self._spec_bins = n_fft // 2 + 1
        # complex head: decoder predicts raw (real, imag) STFT -> ISTFT.
        out_dim = self._spec_bins * 2
        self._proj_in = nn.Conv1d(token_dim, hidden, kernel_size=3, padding=1)
        self._attn = GRUBlock1D(hidden)
        self._res1 = nn.Sequential(
            *[
                Res1DBlock(hidden, num_res_conv, dilation_factor, kernel_size)
                for _ in range(num_res_blocks)
            ]
        )
        self._up: nn.Module = (
            nn.Identity()
            if time_upsample == 1
            else nn.ConvTranspose1d(
                hidden,
                hidden,
                kernel_size=2 * time_upsample,
                stride=time_upsample,
                padding=time_upsample // 2,
            )
        )
        self._res2 = nn.Sequential(
            *[
                Res1DBlock(hidden, num_res_conv, dilation_factor, kernel_size)
                for _ in range(num_res_blocks)
            ]
        )
        self._proj_spec = nn.Conv1d(hidden, out_dim, kernel_size=7, padding=3)
        self._istft = ISTFT(n_fft, hop_length, win_length, padding="same")
        self._end_conv = nn.Sequential(
            nn.Conv1d(1, output_conv_hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(
                output_conv_hidden_dim, output_conv_hidden_dim, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.Conv1d(output_conv_hidden_dim, 1, kernel_size=3, padding=1),
        )

    @property
    def last_layer(self) -> nn.Module:
        return self._end_conv[-1]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, token_dim, T_lat)
        h = self._proj_in(z)
        h = self._attn(h)
        h = self._res1(h)
        h = self._up(h)
        h = self._res2(h)
        spec = self._proj_spec(h)
        # torch.complex/ISTFT don't support bf16 -> fp32 for the spectral head
        spec = spec.float()
        b = self._spec_bins
        complex_spec = torch.complex(spec[:, :b], spec[:, b:])
        y = self._istft(complex_spec)  # (B, L)
        # fp32 output: downstream losses (cuFFT stft, melspec) reject bf16
        return self._end_conv(y.unsqueeze(1)).float()  # (B, 1, L)


# ===========================================================================
# Main VQ-VAE model
# ===========================================================================


class Tokenizer(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def tokenize(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def from_tokens(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, z_e: torch.Tensor, origin_shape=None): ...

    @property
    @abstractmethod
    def last_layer(self) -> nn.Module: ...


class MultiLvlVQVariationalAutoEncoder(Tokenizer):
    def __init__(
        self,
        input_channels: int,
        encoder: nn.Module,
        decoder: DecoderBase,
        vq_module: VQ1D,
        loss_aggregator=None,
        bypass_vq: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.loss_aggregator = loss_aggregator
        self.encoder = encoder
        self.decoder = decoder
        self.vq_module = vq_module
        self.input_channels = input_channels
        # ablation: decoder consumes continuous z_e; VQ still runs (and trains via
        # alignment/commitment) but its output never reaches the decoder
        self.bypass_vq = bypass_vq

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_reshaped = (
            x.reshape((x.shape[0], -1, self.input_channels))
            .permute((0, 2, 1))
            .contiguous()
        )
        # no normalize: unit-sphere latents erase frame energy and make the
        # commitment target (sum of codes, norm != 1) unsatisfiable
        return self.encoder(x_reshaped.float())

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(x)
        return self.vq_module(z_e)["indices"]

    def from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.vq_module.vq_codebook.embed_codebook(indices)
        quantized_outputs = z_q.transpose(1, 2).contiguous()
        return self.decoder(quantized_outputs.transpose(1, 2).contiguous())

    def decode(self, z_e: torch.Tensor, origin_shape=None):
        vq_block_output = self.vq_module(z_e, extract_losses=True)
        dec_in = z_e if self.bypass_vq else vq_block_output["v_q"][:, -1, ...]
        x_out = self.decoder(dec_in)
        if origin_shape is None:
            origin_shape = (int(z_e.shape[0]), self.input_channels, -1)
        x_out = x_out.permute((0, 2, 1)).contiguous().reshape(origin_shape)
        total_output = {**vq_block_output, "slice": x_out}
        return x_out, total_output

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z_e = self.encode(x)
        _, total_output = self.decode(z_e)  # type: ignore
        total_output.update({"z_e": z_e})
        return total_output

    @property
    def last_layer(self) -> nn.Module:
        return self.decoder.last_layer


# ===========================================================================
# Discriminators
# ===========================================================================


class DiscriminatorBase(ABC, nn.Module):
    @property
    def last_layer(self) -> nn.Module:
        return self.class_head.last_layer  # type: ignore


class MelSpecDiscriminator(DiscriminatorBase):
    def __init__(
        self,
        channel_list: list[int],
        stride: int,
        kernel_size: int,
        mel_spec_converter: SimpleMelSpecConverter,
        post_mel_fn: Callable = lambda x: x,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        layers = []
        channel_list = [1] + list(channel_list)
        self._post_mel_fn = post_mel_fn
        for idx in range(len(channel_list) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=channel_list[idx],
                    out_channels=channel_list[idx + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                )
            )
            layers.append(activation_fn)
            layers.append(nn.BatchNorm2d(channel_list[idx + 1]))
        layers.append(
            nn.Conv2d(
                channel_list[-1],
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )
        )
        self.model = nn.Sequential(*layers)
        self._mel_spec_converter = mel_spec_converter

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._mel_spec_converter.convert(x)
        x = self._post_mel_fn(x)
        x = self.model(x)
        x = x.permute((0, 2, 3, 1)).flatten(start_dim=1, end_dim=2).contiguous()
        return {"logits": x}


class WaveformDiscriminator(DiscriminatorBase):
    def __init__(
        self,
        channel_list: list[int],
        dim_change_list: list[int],
        kernel_size: int,
        num_res_block_conv: int,
        dilation_factor: int,
        input_channels: int = 1,
        activation_fn: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        if len(channel_list) != len(dim_change_list) + 1:
            raise ValueError(
                "The length of `channel_list` must be one greater than the length of `dim_change_list`."
            )
        self.last_dim = channel_list[-1]
        self._activation = activation_fn
        self._init_conv = nn.Conv1d(
            input_channels,
            channel_list[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self._conv_list = nn.ModuleList(
            [
                Res1DBlock(
                    channel_list[idx],
                    num_res_block_conv,
                    dilation_factor,
                    kernel_size,
                    activation_fn,
                )
                for idx in range(len(dim_change_list))
            ]
        )
        self._dim_change_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        channel_list[idx],
                        channel_list[idx + 1],
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.MaxPool1d(
                        kernel_size=dim_change_list[idx], stride=dim_change_list[idx]
                    ),
                )
                for idx in range(len(dim_change_list))
            ]
        )
        self._last_conv = nn.Conv1d(
            channel_list[-1], 1, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._init_conv(x)
        for conv, dim_change in zip(self._conv_list, self._dim_change_list):
            x = conv(x)
            x = dim_change(x)
        x = self._last_conv(x)
        return {"logits": x.transpose(1, 2).contiguous()}

    @property
    def last_layer(self) -> nn.Module:
        return self._last_conv


class EnsembleDiscriminator(DiscriminatorBase):
    def __init__(self, discriminators: list) -> None:
        super().__init__()
        self._discriminators = nn.ModuleList(discriminators)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        discriminator_output = [d.forward(x)["logits"] for d in self._discriminators]
        concatenated_output = torch.cat(discriminator_output, dim=1)
        return {"logits": concatenated_output}

    @property
    def last_layer(self) -> nn.Module:
        return self._discriminators[-1].last_layer  # type: ignore


# ===========================================================================
# Lightning modules
# ===========================================================================


class BaseLightningModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_params: LearningParameters,
        transforms: nn.Sequential | None = None,
        loss_aggregator=None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_params = learning_params
        self.loss_aggregator = loss_aggregator
        self.transforms = transforms
        self.optimizer = self._build_optimizer(optimizer_cfg)
        self.scheduler = self._build_scheduler(scheduler_cfg)

    def _build_optimizer(self, optimizer_cfg):
        if optimizer_cfg is not None and optimizer_cfg["target"] != "none":
            filtered = {k: v for k, v in optimizer_cfg.items() if k != "target"}
            target = optimizer_cfg["target"]
            optimizer = getattr(
                importlib.import_module(".".join(target.split(".")[:-1])),
                target.split(".")[-1],
            )(self.parameters(), **filtered)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_params.learning_rate,
                weight_decay=self.learning_params.weight_decay,
                amsgrad=True,
            )
        return optimizer

    def _build_scheduler(self, scheduler_cfg):
        if scheduler_cfg is not None and scheduler_cfg["target"] != "none":
            filtered = {
                k: v
                for k, v in scheduler_cfg.items()
                if k not in ["target", "module_params"]
            }
            target = scheduler_cfg["target"]
            return getattr(
                importlib.import_module(".".join(target.split(".")[:-1])),
                target.split(".")[-1],
            )(self.optimizer, **filtered)
        return None

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.scheduler is None:
            return [self.optimizer]
        scheduler_settings = self._configure_scheduler_settings(
            self.learning_params.interval,
            self.learning_params.loss_monitor,
            self.learning_params.frequency,
        )
        return [self.optimizer], [scheduler_settings]  # type: ignore

    def _configure_scheduler_settings(self, interval, monitor, frequency):
        if self.scheduler is None:
            raise AttributeError("Must include a scheduler")
        return {
            "scheduler": self.scheduler,
            "interval": interval,
            "monitor": monitor,
            "frequency": frequency,
        }

    @abstractmethod
    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        if self.optimizer is None:
            raise AttributeError("For training, an optimizer is required.")
        if self.loss_aggregator is None:
            raise AttributeError("For training, must include a loss aggregator.")
        return self.step(batch, "training")  # type: ignore

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        return self.step(batch, "validation")

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return
        targets = {
            "z_e": output["z_e"],
            "slice": batch["slice"],
            "class": quantize_waveform_256(batch["slice"]).long(),
        }
        loss = self.loss_aggregator(output, targets)
        for ind_loss, value in loss.individual.items():
            self.log(
                f"test/{ind_loss}",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.learning_params.batch_size,
            )
        self.log(
            "test/total",
            loss.total,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
        )
        if not hasattr(self, "_cdpam_fn"):
            self._cdpam_fn = make_cdpam_evaluator(str(self.device))
        cdpam_score = cdpam_distance(
            self._cdpam_fn, output["slice"].detach(), batch["slice"]
        )
        self.log(
            "test/cdpam",
            cdpam_score,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
        )
        # Composite headline (lower=better): cdpam (timbre) + mrstft (spectral) +
        # chroma (melody). mrstft/chroma already logged via loss.individual above.
        w = COMPOSITE_WEIGHTS
        composite = (
            w["cdpam"] * cdpam_score
            + w["mrstft"] * loss.individual["mrstft"]
            + w["chroma"] * loss.individual["chroma"]
            + w["phase"] * loss.individual["phase"]
        )
        self.log(
            "test/composite",
            composite,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
        )

    @abstractmethod
    def step(self, batch, phase) -> torch.Tensor | None: ...


class MusicLightningModule(BaseLightningModule):
    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(input["slice"])

    def handle_loss(self, loss: LossOutput, phase: str) -> torch.Tensor:
        for name in loss.individual:
            self.log(
                f"{phase}/{name.replace('_', ' ')}",
                loss.individual[name],
                sync_dist=True,
                batch_size=self.learning_params.batch_size,
            )
        self.log(
            f"{phase}/total loss",
            loss.total,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
        )
        return loss.total

    def step(self, batch, phase) -> torch.Tensor | None:
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return None
        targets = {
            "z_e": output["z_e"],
            "slice": batch["slice"],
            "class": quantize_waveform_256(batch["slice"]).long(),
        }
        loss = self.loss_aggregator(output, targets)
        return self.handle_loss(loss, phase)

    def on_train_epoch_end(self) -> None:
        if hasattr(self.model, "vq_module"):
            num_dead_codes = self.model.vq_module.vq_codebook.random_restart()  # type: ignore
            self.model.vq_module.vq_codebook.reset_usage()  # type: ignore
            self.log(
                "number of dead codes",
                num_dead_codes,
                sync_dist=True,
                batch_size=self.learning_params.batch_size,
            )


class ModelPart(Enum):
    GENERATOR = "generator"
    DISCRIMINATOR = "discriminator"


class VqganMusicLightningModule(MusicLightningModule):
    def __init__(
        self,
        model: MultiLvlVQVariationalAutoEncoder,
        discriminator: EnsembleDiscriminator,
        learning_params: LearningParameters,
        discriminator_loss,
        generator_loss,
        generator_start_step: int = 10000,
        disc_warmup: int = 0,
        d_weight_cap: float = 1e4,
        transforms: nn.Sequential | None = None,
        loss_aggregator=None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model,
            learning_params,
            transforms,
            loss_aggregator,
            optimizer_cfg,
            scheduler_cfg,
        )
        self.discriminator = discriminator
        self._generator_start_step = generator_start_step
        # Disc trains alone for `disc_warmup` steps before generator gets the adversarial
        # gradient — gives the cold discriminator a head start vs the converged generator.
        self._disc_warmup = disc_warmup
        # Upper clamp on the adaptive adversarial weight. The raw ratio
        # ‖∇rec‖/‖∇gen‖ runs away to a huge ceiling once gen grads shrink, which
        # dumps an overwhelming adversarial gradient that wrecks reconstruction
        # (observed: dw4_long regressed all metrics). Cap low (~1) to keep the
        # GAN as a texture nudge rather than the dominant objective.
        self._d_weight_cap = d_weight_cap
        self._adv_start = generator_start_step + disc_warmup
        self.optimizer_d = self._build_optimizer_gan(
            optimizer_cfg, ModelPart.DISCRIMINATOR
        )
        self.optimizer_g = self._build_optimizer_gan(optimizer_cfg, ModelPart.GENERATOR)
        self.scheduler_d = self._build_scheduler_gan(
            scheduler_cfg, ModelPart.DISCRIMINATOR
        )
        self.scheduler_g = self._build_scheduler_gan(scheduler_cfg, ModelPart.GENERATOR)
        self._current_step = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.automatic_optimization = False
        self._discriminator_loss = discriminator_loss
        self._generator_loss = generator_loss
        self.optimizer = 0
        del self.scheduler

    @property
    def generator(self) -> MultiLvlVQVariationalAutoEncoder:
        return self.model  # type: ignore

    def _build_optimizer_gan(
        self, optimizer_cfg, model_part: ModelPart
    ) -> torch.optim.Optimizer:
        if model_part == ModelPart.GENERATOR:
            parameters = self.model.parameters()
        elif model_part == ModelPart.DISCRIMINATOR:
            parameters = self.discriminator.parameters()
        else:
            raise ValueError(f"Invalid model part {model_part}")

        if optimizer_cfg is not None and optimizer_cfg["target"] != "none":
            filtered = {k: v for k, v in optimizer_cfg.items() if k != "target"}
            target = optimizer_cfg["target"]
            optimizer = getattr(
                importlib.import_module(".".join(target.split(".")[:-1])),
                target.split(".")[-1],
            )(parameters, **filtered)
        else:
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.learning_params.learning_rate,
                weight_decay=self.learning_params.weight_decay,
                amsgrad=True,
            )
        return optimizer

    def _build_scheduler_gan(self, scheduler_cfg, model_part: ModelPart):
        if scheduler_cfg is None or scheduler_cfg["target"] == "none":
            return None
        match model_part:
            case ModelPart.GENERATOR:
                optimizer = self.optimizer_g
            case ModelPart.DISCRIMINATOR:
                optimizer = self.optimizer_d
            case _:
                raise ValueError(f"Invalid model part {model_part}")
        filtered = {
            k: v
            for k, v in scheduler_cfg.items()
            if k not in ["target", "module_params"]
        }
        target = scheduler_cfg["target"]
        return getattr(
            importlib.import_module(".".join(target.split(".")[:-1])),
            target.split(".")[-1],
        )(optimizer, **filtered)

    def calculate_adaptive_weight(
        self, rec_loss, generator_loss, last_layer: nn.Conv1d
    ):
        rec_grads = torch.autograd.grad(rec_loss, last_layer.weight, retain_graph=True)[
            0
        ]
        generator_grads = torch.autograd.grad(
            generator_loss, last_layer.weight, retain_graph=True
        )[0]
        d_weight = torch.norm(rec_grads) / (torch.norm(generator_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self._d_weight_cap).detach()
        return d_weight

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.scheduler_d is None or self.scheduler_g is None:
            return [self.optimizer_d, self.optimizer_g], []
        scheduler_settings_g = {
            "scheduler": self.scheduler_g,
            "interval": self.learning_params.interval,
            "monitor": self.learning_params.loss_monitor,
            "frequency": self.learning_params.frequency,
        }
        scheduler_settings_d = {
            "scheduler": self.scheduler_d,
            "interval": self.learning_params.interval,
            "monitor": self.learning_params.loss_monitor,
            "frequency": self.learning_params.frequency,
        }
        self.optimizer = self.optimizer_g
        return [self.optimizer_d, self.optimizer_g], [scheduler_settings_d, scheduler_settings_g]  # type: ignore

    def step(self, batch, phase) -> torch.Tensor | None:
        if phase == "training":
            self._current_step.data = self._current_step.data + 1

        optimizer_d, optimizer_g = self.optimizers()  # type: ignore
        scheduler_d, scheduler_g = None, None
        scheduler_list = self.lr_schedulers()
        if scheduler_list is not None and len(scheduler_list) > 0:  # type: ignore
            scheduler_d, scheduler_g = scheduler_list  # type: ignore

        if self._current_step >= self._generator_start_step or phase != "training":
            self._discriminator_step(batch, phase, optimizer_d, scheduler_d)
        return self._generator_step(batch, phase, optimizer_g, scheduler_g)

    def _close_generator_phase(
        self, phase, loss, d_weight, generator_loss, optimizer_g
    ):
        self.log(
            f"{phase}/generator loss",
            generator_loss / (d_weight + 1e-6),
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
        )
        self.handle_loss(loss, phase)
        self.untoggle_optimizer(optimizer_g)

    def _discriminator_step(self, batch, phase, optimizer_d, scheduler_d):
        self.toggle_optimizer(optimizer_d)
        disc_outputs_real = self.discriminator(batch["slice"])
        with torch.no_grad():
            restructured_outputs_frozen = self.forward(batch)
            fake_slice = restructured_outputs_frozen["slice"].detach()
        disc_outputs_fake = self.discriminator(fake_slice.requires_grad_())
        disc_outputs = {
            "d_input": disc_outputs_real["logits"],
            "d_output": disc_outputs_fake["logits"],
        }
        disc_outputs.update(restructured_outputs_frozen)
        disc_loss = self._discriminator_loss(disc_outputs, {})
        if phase == "training" and self._current_step >= self._generator_start_step:
            self.manual_backward(disc_loss * self._discriminator_loss.weight)
            self.clip_gradients(
                optimizer_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            optimizer_d.step()
            optimizer_d.zero_grad()
        self.log(
            f"{phase}/discriminator loss",
            disc_loss if self._current_step >= self._generator_start_step else 0,
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
        )
        if scheduler_d is not None and phase == "training":
            scheduler_d.step()  # type: ignore
        self.untoggle_optimizer(optimizer_d)

    def _generator_step(
        self, batch, phase, optimizer_g, scheduler_g
    ) -> torch.Tensor | None:
        self.toggle_optimizer(optimizer_g)
        restructured_outputs = self.forward(batch)
        gan_active = self._current_step >= self._adv_start or phase != "training"
        if gan_active:
            disc_outputs_fake = self.discriminator(restructured_outputs["slice"])
            restructured_outputs["d_output"] = disc_outputs_fake["logits"]
            disc_outputs_real = self.discriminator(batch["slice"])
            restructured_outputs["d_input"] = disc_outputs_real["logits"]
        targets = {
            "z_e": restructured_outputs["z_e"],
            "slice": batch["slice"],
            "class": quantize_waveform_256(batch["slice"]).long(),
        }
        if self.loss_aggregator is None:
            self.untoggle_optimizer(optimizer_g)
            return None

        if phase == "training":
            # diagnostic: z_e scale drift drives align/commit MSE growth
            self.log(
                "training/z_e rms",
                restructured_outputs["z_e"].detach().pow(2).mean().sqrt(),
                sync_dist=True,
                batch_size=self.learning_params.batch_size,
            )

        loss = self.loss_aggregator(restructured_outputs, targets)
        generator_loss = (
            self._generator_loss(restructured_outputs, {}) if gan_active else 0
        )

        if phase != "training":
            self._close_generator_phase(
                phase=phase,
                loss=loss,
                d_weight=1.0,
                generator_loss=generator_loss,
                optimizer_g=optimizer_g,
            )
            return loss.total.clone()

        if self._current_step >= self._adv_start:
            d_weight = self.calculate_adaptive_weight(
                loss.total, generator_loss, self.generator.last_layer  # type: ignore
            )
            generator_loss = generator_loss * d_weight
            self.log(
                "d_weight",
                d_weight,
                sync_dist=True,
                batch_size=self.learning_params.batch_size,
            )
            self.manual_backward(
                loss.total + generator_loss * self._generator_loss.weight
            )
        else:
            generator_loss = 0
            d_weight = 0
            self.log("d_weight", 0, sync_dist=True)
            self.manual_backward(loss.total)

        self.clip_gradients(
            optimizer_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        optimizer_g.step()
        optimizer_g.zero_grad()
        if scheduler_g is not None and phase == "training":
            scheduler_g.step()  # type: ignore

        self._close_generator_phase(
            phase=phase,
            loss=loss,
            d_weight=d_weight if phase == "training" else 1.0,
            generator_loss=generator_loss,
            optimizer_g=optimizer_g,
        )
        return loss.total.clone()


# ===========================================================================
# Build everything from hardcoded lvl1_vqgan config
# ===========================================================================


def build_learning_params() -> LearningParameters:
    return LearningParameters(
        model_name="lvl1_vqgan",
        learning_rate=0.0001,
        weight_decay=0.0,
        batch_size=16,
        grad_accumulation=1,
        epochs=10000,
        beta_ema=0.95,
        gradient_clip=None,
        save_path="saved/",
        amp=True,
        val_split=0.04,
        test_split=0.01,
        devices="auto",
        num_workers=8,
        loss_monitor="validation/total loss",
        trigger_loss=1e-8,
        interval="step",
        frequency=1,
    )


def build_optimizer_cfg() -> dict[str, Any]:
    return {
        "target": "torch.optim.AdamW",
        "weight_decay": 0.0,
        "lr": 0.0001,
        "amsgrad": True,
        "betas": (0.5, 0.95),
    }


class TimeOneCycleLR(torch.optim.lr_scheduler.LRScheduler):
    """Wall-clock one-cycle LR, the time-capped analogue of torch's OneCycleLR.

    OneCycleLR is step-indexed and raises once stepped past `total_steps`; training
    here is time-capped (Timer(minutes=...)) with unknown step count, so we drive
    the cycle by elapsed wall time vs `total_minutes`.

    Shape (anneal_strategy='cos', matching torch defaults):
        warmup  (frac < pct_start): initial_lr -> max_lr   over the first pct_start
        anneal  (frac >= pct_start): max_lr     -> min_lr   over the remainder
    Default pct_start=0.1 (rise over first 10% of training time, then anneal),
    with initial_lr = max_lr/div_factor and min_lr = initial_lr/final_div_factor.
    Clock starts on the first get_lr() (start of training); on resume it restarts,
    so set minutes to the resume duration.
    """

    def __init__(
        self,
        optimizer,
        total_minutes: float,
        max_lr: float,
        pct_start: float = 0.1,
        div_factor: float = 10.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        self.total_seconds = max(float(total_minutes) * 60.0, 1e-8)
        self.max_lr = float(max_lr)
        self.pct_start = float(pct_start)
        self.initial_lr = self.max_lr / float(div_factor)
        self.min_lr = self.initial_lr / float(final_div_factor)
        self._start: float | None = None
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        import math
        import time

        if self._start is None:
            self._start = time.monotonic()
        frac = min((time.monotonic() - self._start) / self.total_seconds, 1.0)
        if frac < self.pct_start:
            # warmup: initial_lr -> max_lr, cosine ease-in
            p = frac / self.pct_start
            lr = (
                self.initial_lr
                + (self.max_lr - self.initial_lr) * (1.0 - math.cos(math.pi * p)) / 2.0
            )
        else:
            # anneal: max_lr -> min_lr, cosine ease-out
            p = (frac - self.pct_start) / max(1.0 - self.pct_start, 1e-8)
            lr = (
                self.min_lr
                + (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * p)) / 2.0
            )
        return [lr for _ in self.base_lrs]


def build_scheduler_cfg(
    total_minutes: float = 540.0,
    max_lr: float = 1e-4,
    pct_start: float = 0.1,
) -> dict[str, Any]:
    module_params = {
        "loss_monitor": "training/total loss",
        "interval": "step",
        "frequency": 1,
        "trigger_loss": 1e-8,
    }
    return {
        # resolve from the running module (__main__ when run as a script, else
        # "train") so we don't re-import train.py as a second module.
        "target": f"{__name__}.TimeOneCycleLR",
        "total_minutes": total_minutes,
        "max_lr": max_lr,
        "pct_start": pct_start,
        "module_params": module_params,
    }


def build_loss_aggregator(commit_weight: float = 0.75) -> WeightedSumAggregator:
    def make_mel_converter(
        n_fft, hop_length, n_mels, power, norm="slaney", mel_scale="htk"
    ) -> SimpleMelSpecConverter:
        return SimpleMelSpecConverter(
            MelSpecParameters(
                n_fft=n_fft,
                f_min=0,
                hop_length=hop_length,
                n_mels=n_mels,
                pad_mode="reflect",
                power=power,
                norm=norm,
                mel_scale=mel_scale,
                pad=0,
            )
        )

    components = [
        RecLoss(
            name="rec_l2",
            weight=0.15,
            pred_key="slice",
            ref_key="slice",
            base_loss=nn.MSELoss(),
        ),
        EdgeRecLoss(
            name="edge_rec",
            weight=1.0,
            pred_key="slice",
            ref_key="slice",
            edge_power=3.0,
            base_loss=nn.MSELoss(),
        ),
        AlignLoss(name="alignment_loss", weight=0.75, base_loss=nn.MSELoss()),
        # commit_weight anchors unnormalized z_e to the codebook; 0.75 was too weak
        # (z_e magnitude outran codes -> alignment drifted unbounded over 3h runs)
        CommitLoss(
            name="commitment_loss", weight=commit_weight, base_loss=nn.MSELoss()
        ),
        MelSpecLoss(
            name="melspec_loss_2",
            weight=2.0,
            pred_key="slice",
            ref_key="slice",
            base_loss=nn.L1Loss(),
            lin_start=0.5,
            lin_end=5.0,
            mel_spec_converter=make_mel_converter(1024, 256, 64, 2.0),
        ),
        MelSpecLoss(
            name="melspec_loss_3",
            weight=5.0,
            pred_key="slice",
            ref_key="slice",
            base_loss=nn.L1Loss(),
            lin_start=1.0,
            lin_end=20.0,
            transform_func=transparent,
            mel_spec_converter=make_mel_converter(512, 128, 32, 1.0),
        ),
        MelSpecLoss(
            name="melspec_loss_4",
            weight=10.0,
            pred_key="slice",
            ref_key="slice",
            base_loss=nn.L1Loss(),
            lin_start=0.5,
            lin_end=10.0,
            mel_spec_converter=make_mel_converter(4096, 1024, 256, 1.0),
        ),
        MelSpecLoss(
            name="melspec_loss_6",
            weight=5.0,
            pred_key="slice",
            ref_key="slice",
            base_loss=nn.L1Loss(),
            lin_start=1.0,
            lin_end=1.0,
            transform_func=transparent,
            mel_spec_converter=make_mel_converter(4096, 1024, 256, 1.0),
        ),
        CDPAMLoss(
            name="cdpam",
            weight=25.0,
            pred_key="slice",
            ref_key="slice",
        ),
        # Overall spectral fidelity (rhythm + harmonics). Starting weight; tune.
        MultiResSTFTLoss(
            name="mrstft",
            weight=5.0,
            pred_key="slice",
            ref_key="slice",
        ),
        # MELODY term — weighted up since pitch is the current weak spot. Tune.
        ChromaLoss(
            name="chroma",
            weight=10.0,
            pred_key="slice",
            ref_key="slice",
        ),
        # PHASE coherence — DIAGNOSTIC ONLY (weight 0). Absolute-phase cos-distance is
        # ill-posed (time-shift variant), proven ineffective as a loss; GAN handles phase.
        PhaseLoss(
            name="phase",
            weight=0.0,
            pred_key="slice",
            ref_key="slice",
        ),
    ]
    return WeightedSumAggregator(components)


def build_vq_module(
    token_dim: int = 512, num_rq_steps: int = 3, num_tokens: int = 2048
) -> VQ1D:
    return VQ1D(token_dim=token_dim, num_tokens=num_tokens, num_rq_steps=num_rq_steps)


def build_generator(
    loss_aggregator: WeightedSumAggregator,
    token_dim: int = 1024,
    num_rq_steps: int = 3,
    num_tokens: int = 2048,
    time_downsample: int = 1,
    hidden: int = 1024,
) -> MultiLvlVQVariationalAutoEncoder:
    # temporal STFT arch: 32768-sample slice, hop 256 -> 128 STFT frames ->
    # T_lat = 128 / time_downsample. Encoder l2-normalizes z_e; decoder is the
    # complex (real/imag) STFT head.
    encoder = TemporalEncoder(
        token_dim=token_dim, hidden=hidden, time_downsample=time_downsample
    )
    decoder = TemporalDecoder(
        token_dim=token_dim,
        hidden=hidden,
        num_res_blocks=3,
        time_upsample=time_downsample,
    )
    return MultiLvlVQVariationalAutoEncoder(
        input_channels=1,
        encoder=encoder,
        decoder=decoder,
        vq_module=build_vq_module(token_dim, num_rq_steps, num_tokens),
        loss_aggregator=loss_aggregator,
    )


def _spectralize_disc(module: nn.Module) -> nn.Module:
    """Discriminator hygiene: spectral-norm every conv (1-Lipschitz control, the
    principled stabilizer) and drop BatchNorm (an anti-pattern in GAN discriminators —
    leaks batch statistics across real/fake and fights the adversarial signal).
    Recurses, mutates in place. Applied only to discriminators, never the generator."""
    from torch.nn.utils.parametrizations import spectral_norm

    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            setattr(module, name, nn.Identity())
        elif isinstance(child, (nn.Conv1d, nn.Conv2d)):
            setattr(module, name, spectral_norm(child))
        else:
            _spectralize_disc(child)
    return module


def build_discriminator(disc_width: int = 1) -> EnsembleDiscriminator:
    w = disc_width

    def make_mel_conv(n_fft, hop_length, n_mels, power) -> SimpleMelSpecConverter:
        return SimpleMelSpecConverter(
            MelSpecParameters(
                n_fft=n_fft,
                f_min=0,
                hop_length=hop_length,
                n_mels=n_mels,
                pad_mode="reflect",
                power=power,
                norm="slaney",
                mel_scale="htk",
                pad=0,
            )
        )

    disc1 = MelSpecDiscriminator(
        channel_list=[4 * w, 8 * w, 16 * w, 32 * w, 64 * w],
        stride=2,
        kernel_size=3,
        activation_fn=nn.GELU(),
        post_mel_fn=log_normal,
        mel_spec_converter=make_mel_conv(1024, 256, 128, 1.0),
    )
    disc2 = MelSpecDiscriminator(
        channel_list=[4 * w, 8 * w, 16 * w, 32 * w, 64 * w],
        stride=2,
        kernel_size=3,
        activation_fn=nn.GELU(),
        post_mel_fn=transparent,
        mel_spec_converter=make_mel_conv(2048, 512, 256, 1.0),
    )
    disc3 = MelSpecDiscriminator(
        channel_list=[4 * w, 8 * w, 16 * w, 32 * w, 64 * w],
        stride=2,
        kernel_size=3,
        activation_fn=nn.GELU(),
        post_mel_fn=tanh_fn,
        mel_spec_converter=make_mel_conv(4096, 1024, 512, 1.0),
    )
    disc4 = WaveformDiscriminator(
        channel_list=[4 * w, 8 * w, 16 * w, 32 * w, 64 * w, 128 * w],
        dim_change_list=[4, 4, 4, 4, 4],
        kernel_size=3,
        num_res_block_conv=5,
        dilation_factor=3,
        input_channels=1,
        activation_fn=nn.GELU(),
    )
    return EnsembleDiscriminator(
        [_spectralize_disc(d) for d in (disc1, disc2, disc3, disc4)]
    )


def build_module(
    learning_params: LearningParameters,
    loss_aggregator: WeightedSumAggregator,
    optimizer_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    gss: int = 20000,
    token_dim: int = 1024,
    num_rq_steps: int = 3,
    num_tokens: int = 2048,
    time_downsample: int = 1,
    disc_width: int = 1,
    d_weight_cap: float = 1.0,
    hidden: int = 1024,
) -> VqganMusicLightningModule:
    generator = build_generator(
        loss_aggregator,
        token_dim=token_dim,
        num_rq_steps=num_rq_steps,
        num_tokens=num_tokens,
        time_downsample=time_downsample,
        hidden=hidden,
    )
    discriminator = build_discriminator(disc_width=disc_width)

    discriminator_loss = DiscriminatorHingeLoss(
        name="discriminator_loss",
        weight=1.0,
        pred_key_real="d_input",
        pred_key_fake="d_output",
    )
    generator_loss = GeneratorHingeLoss(
        name="generator_loss",
        weight=1.0,
        pred_key_disc="d_output",
    )

    module = VqganMusicLightningModule(
        model=generator,
        discriminator=discriminator,
        learning_params=learning_params,
        discriminator_loss=discriminator_loss,
        generator_loss=generator_loss,
        generator_start_step=gss,
        disc_warmup=0,
        d_weight_cap=d_weight_cap,
        loss_aggregator=loss_aggregator,
        optimizer_cfg=optimizer_cfg,
        scheduler_cfg=scheduler_cfg,
    )
    return module


# ===========================================================================
# Entry point
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lvl1_vqgan.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to the YAML config (default: config.yaml next to train.py).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    m, vq, gan, tr = cfg["model"], cfg["vq"], cfg["gan"], cfg["train"]

    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    torch.set_float32_matmul_precision("high")
    L.seed_everything(42, workers=True)

    learning_params = build_learning_params()
    if tr.get("save_path") is not None:
        learning_params.save_path = tr["save_path"]
    learning_params.devices = tr.get("devices", "auto")
    learning_params.learning_rate = tr["lr"]
    optimizer_cfg = build_optimizer_cfg()
    # the GAN optimizers build from optimizer_cfg, so its "lr" is the one that
    # takes effect (and the one-cycle peak) -- set it from config.
    optimizer_cfg["lr"] = tr["lr"]
    scheduler_cfg = build_scheduler_cfg(
        total_minutes=tr["minutes"],
        max_lr=tr["lr"],
        pct_start=tr["lr_pct_start"],
    )
    loss_aggregator = build_loss_aggregator(commit_weight=vq["commit_weight"])

    print("Initializing data...")
    data_module = build_data_module(learning_params)

    single_song = tr.get("single_song")
    if single_song is not None:
        dataset = data_module._dataset  # type: ignore[attr-defined]
        before = len(dataset.buffer)
        dataset.buffer = [dp for dp in dataset.buffer if single_song in dp.track_path]
        print(
            f"single-song filter '{single_song}': {before} -> {len(dataset.buffer)} slices"
        )
        if len(dataset.buffer) == 0:
            raise ValueError(f"No tracks matched single_song '{single_song}'")

    print("Initializing trainer...")
    trainer = get_trainer(learning_params, minutes=tr["minutes"])

    print("Initializing module...")
    module = build_module(
        learning_params,
        loss_aggregator,
        optimizer_cfg,
        scheduler_cfg,
        gss=gan["gan_start_step"],
        token_dim=m["token_dim"],
        num_rq_steps=m["num_rq"],
        num_tokens=m["num_tokens"],
        time_downsample=m["time_downsample"],
        disc_width=gan["disc_width"],
        d_weight_cap=gan["d_weight_cap"],
        hidden=m["hidden"],
    )

    checkpoint = tr.get("checkpoint")
    if checkpoint is not None:
        print(f"Loading model weights from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # strict=False so the generator can resume into a fresh discriminator.
        result = module.load_state_dict(state_dict, strict=False)
        miss = [k for k in result.missing_keys if not k.startswith("discriminator")]
        unexp = [k for k in result.unexpected_keys if not k.startswith("discriminator")]
        print(
            f"  loaded; discriminator left at init. "
            f"non-disc missing={len(miss)} unexpected={len(unexp)}"
        )
        if miss or unexp:
            print(
                f"  WARN non-disc mismatch: missing={miss[:5]} unexpected={unexp[:5]}"
            )

    print("Starting training...")
    trainer.fit(module, data_module)
    print("Training complete.")

    print("\nRunning test evaluation...")
    results = trainer.test(module, data_module)
    print("Test complete. Results printed above.")

    cdpam_val = results[0].get("test/cdpam", 0.0) if results else 0.0
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    print(f"cdpam: {cdpam_val:.6f}")
    print(f"peak_vram_mb: {peak_vram_mb:.0f}")


if __name__ == "__main__":
    main()
