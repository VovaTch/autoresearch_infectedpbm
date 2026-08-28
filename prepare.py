"""
Data preparation: dataset, dataloaders, and mel spectrogram converters for lvl1_vqgan training.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio import load
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader, Dataset, random_split

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import cdpam as _cdpam_lib
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

SLICE_DATASET_SIZE = 0


@dataclass
class LearningParameters:
    model_name: str
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    batch_size: int = 32
    grad_accumulation: int = 1
    epochs: int = 10
    beta_ema: float = 0.999
    gradient_clip: float | None = 0.5
    save_path: str = "saved"
    amp: bool = False
    val_split: float = 0.05
    test_split: float = 0.01
    devices: Any = "auto"
    num_workers: int = 0
    loss_monitor: str = "validation/total loss"
    trigger_loss: float = 0.0
    interval: str = "step"
    frequency: int = 1
    limit_train_batches: int | float | None = None
    limit_eval_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    pin_memory: bool = True


@dataclass
class MelSpecParameters:
    n_fft: int
    hop_length: int
    n_mels: int
    power: float
    f_min: float
    pad: int
    pad_mode: str = "reflect"
    norm: str = "slaney"
    mel_scale: str = "htk"


# ---------------------------------------------------------------------------
# Mel spectrogram converters
# ---------------------------------------------------------------------------


class MelSpecConverter(ABC):
    mel_spec: MelSpectrogram

    @abstractmethod
    def __init__(self, mel_spec_params: MelSpecParameters) -> None: ...

    @abstractmethod
    def convert(self, slice: torch.Tensor) -> torch.Tensor: ...


class SimpleMelSpecConverter(MelSpecConverter):
    def __init__(self, mel_spec_params: MelSpecParameters) -> None:
        self.mel_spec_params = mel_spec_params
        self.mel_spec = MelSpectrogram(**asdict(mel_spec_params))

    def convert(self, slice: torch.Tensor) -> torch.Tensor:
        # Audio front-end stays fp32: cuFFT has no bf16, and a bf16 Trainer
        # precision casts this module's window + mel filterbank buffers (it is
        # reachable from the discriminator, which IS a LightningModule child).
        self.mel_spec = self.mel_spec.to(device=slice.device, dtype=torch.float32)
        return self.mel_spec(slice.float())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class DataPoint:
    _id: int
    slice: torch.Tensor | None
    slice_file_path: str | None
    track_path: str
    track_idx: int
    slice_idx: int
    slice_init_idx: int
    slice_init_time: float

    def get_metadata(self) -> dict[str, Any]:
        return {
            "_id": self._id,
            "slice_file_path": self.slice_file_path,
            "track_path": self.track_path,
            "track_idx": self.track_idx,
            "slice_idx": self.slice_idx,
            "slice_init_idx": self.slice_init_idx,
            "slice_init_time": self.slice_init_time,
        }


class MP3SliceDataset(Dataset):
    buffer: list[DataPoint]

    def __init__(
        self,
        data_path: str,
        sample_rate: int,
        slice_length: int,
        device: str = "cpu",
        processed_path: str | None = None,
        save_processed: bool = False,
        use_preloaded: bool = False,
        slice_dataset_size: int = 0,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._data_path = data_path
        self._slice_length = slice_length
        self._device = device
        self._file_cache: dict[str, torch.Tensor] = {}
        self._slice_dataset_size = slice_dataset_size

        if processed_path and os.path.exists(processed_path):
            logger.info(f"Loading processed data from {processed_path}...")
            self.buffer = self._load_data(processed_path, use_preloaded)
            logger.info("Loaded processed data")
            return

        logger.info(f"Generating data from {data_path}...")
        self.buffer = self._generate_data(data_path, processed_path)
        logger.info("Generated data")

        if save_processed and processed_path:
            os.makedirs(processed_path, exist_ok=True)
            self._dump_data(processed_path)
            logger.info(f"Saved processed data to {processed_path}")

    def _generate_data(
        self, data_path: str, processed_path: str | None
    ) -> list[DataPoint]:
        self.file_list = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".mp3"):
                    self.file_list.append(os.path.join(root, file))
        self.file_list = sorted(self.file_list)

        running_idx = 0
        data_collection = []
        for idx, file in tqdm.tqdm(
            enumerate(self.file_list), "Loading data from files..."
        ):
            file_data = self._load_data_from_track(
                file, idx, processed_path, running_idx
            )
            running_idx += len(file_data)
            data_collection += file_data
        return data_collection

    def _load_data_from_track(
        self, file: str, track_idx: int, processed_path: str | None, running_idx: int
    ) -> list[DataPoint]:
        print(f"loading {file}")
        long_data, sr = load(file, format="mp3")  # type: ignore
        long_data = self._resample_if_necessary(long_data, sr)
        long_data = self._mix_down_if_necessary(long_data)
        long_data = self._right_pad_if_necessary(long_data)
        slices = long_data.view((-1, 1, self._slice_length)).contiguous()
        slice_file_name = self._get_slices_file_name_from_path(file, processed_path)

        data_points: list[DataPoint] = []
        for idx in range(slices.shape[0]):
            data_point = DataPoint(
                _id=running_idx + idx,
                slice=slices[idx].to(self._device),
                slice_file_path=slice_file_name,
                track_path=file,
                track_idx=track_idx,
                slice_idx=idx,
                slice_init_idx=idx * self._slice_length,
                slice_init_time=idx * self._slice_length / self._sample_rate,
            )
            data_points.append(data_point)
        return data_points

    @staticmethod
    def _get_slices_file_name_from_path(
        file_path: str, processed_path: str | None
    ) -> str | None:
        if not processed_path:
            return None
        _, file = os.path.split(file_path)
        name, _ = os.path.splitext(file)
        new_filename = f"slices_{name}.pt".replace(" ", "_")
        return os.path.join(processed_path, new_filename)

    def _resample_if_necessary(
        self, signal: torch.Tensor, sampling_rate: int
    ) -> torch.Tensor:
        if sampling_rate != self._sample_rate:
            resampler = torchaudio.transforms.Resample(sampling_rate, self._sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _right_pad_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        length_signal = signal.shape[1]
        if length_signal % self._slice_length != 0:
            num_missing_samples = (
                self._slice_length - length_signal % self._slice_length
            )
            signal = F.pad(signal, (0, num_missing_samples))
        return signal

    def _dump_data(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        metadata_path = os.path.join(path, "_metadata.json")
        metadata_for_json = [buffer_data.get_metadata() for buffer_data in self.buffer]
        with open(metadata_path, "w") as f:
            json.dump(metadata_for_json, f, indent=4)

        file_mapping = {
            buffer_data.track_path: buffer_data.slice_file_path
            for buffer_data in self.buffer
        }
        for track_file, slice_file in tqdm.tqdm(
            file_mapping.items(), "Saving slices as .pt files..."
        ):
            extracted_data_points = list(
                filter(lambda x: x.track_path == track_file, self.buffer)
            )
            sorted_data_points = sorted(
                extracted_data_points, key=lambda x: x.slice_idx
            )
            aggregate_slices = torch.stack(
                [dp.slice for dp in sorted_data_points], dim=0  # type: ignore
            )
            if slice_file is not None:
                torch.save(aggregate_slices, slice_file)

    def _load_data(self, path: str, use_preloaded: bool = True) -> list[DataPoint]:
        loaded_data = []
        json_file_path = os.path.join(path, "_metadata.json")
        with open(json_file_path, "r") as f:
            metadata: list[dict[str, Any]] = json.load(f)

        for dp in metadata:
            dp["track_path"] = os.path.expanduser(dp["track_path"])
            if dp.get("slice_file_path"):
                dp["slice_file_path"] = os.path.expanduser(dp["slice_file_path"])

        if not use_preloaded:
            for slice_data_point in metadata:
                loaded_data.append(DataPoint(**{**slice_data_point, "slice": None}))  # type: ignore
            logger.info("Parsed metadata to the buffer (lazy mode)")
            return loaded_data

        slice_files: list[str] = list(
            sorted({dp["slice_file_path"] for dp in metadata})
        )
        for slice_file in tqdm.tqdm(slice_files, "Loading slices..."):
            loaded_slice: torch.Tensor = torch.load(slice_file)
            slice_metadata = list(
                filter(lambda x: x["slice_file_path"] == slice_file, metadata)
            )
            slice_metadata = sorted(slice_metadata, key=lambda x: x["slice_idx"])
            for idx, slice_data_point in enumerate(slice_metadata):
                new_data_point = DataPoint(**{**slice_data_point, "slice": loaded_slice[idx, ...]})  # type: ignore
                loaded_data.append(new_data_point)

        logger.info("Parsed metadata and the slices to the buffer")
        return loaded_data

    def __getitem__(self, index: int) -> dict[str, Any]:
        data_point = self.buffer[index]
        if data_point.slice is None:
            assert (
                data_point.slice_file_path is not None
            ), f"Missing sample for sample {index}: {data_point.slice_idx}"
            fp = data_point.slice_file_path
            if fp not in self._file_cache:
                self._file_cache[fp] = torch.load(fp, mmap=True)
            slice_tensor = self._file_cache[fp][data_point.slice_idx].clone()
        else:
            slice_tensor = data_point.slice
        return {**data_point.get_metadata(), "slice": slice_tensor.to(self._device)}

    def __len__(self) -> int:
        if self._slice_dataset_size == 0:
            return len(self.buffer)
        else:
            return min(len(self.buffer), self._slice_dataset_size)


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------


class SplitDatasetModule(L.LightningDataModule):
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def __init__(self, learning_params: LearningParameters, dataset: Dataset) -> None:
        super().__init__()
        self._learning_params = learning_params
        self._dataset = dataset

    def setup(self, stage: str) -> None:
        if self._learning_params.test_split + self._learning_params.val_split > 1:
            raise ValueError("Sum of test and validation split must be <= 1")
        if self._learning_params.test_split < 0 or self._learning_params.val_split < 0:
            raise ValueError("Test and validation split must not be negative.")

        training_len = int(
            (1 - self._learning_params.val_split - self._learning_params.test_split)
            * len(self._dataset)  # type: ignore
        )
        test_len = int(training_len * self._learning_params.test_split)
        val_len = len(self._dataset) - training_len - test_len  # type: ignore

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self._dataset, lengths=(training_len, val_len, test_len)
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self._learning_params.batch_size,
            shuffle=True,
            num_workers=self._learning_params.num_workers,
            pin_memory=self._learning_params.pin_memory,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self._learning_params.batch_size,
            shuffle=False,
            num_workers=self._learning_params.num_workers,
            pin_memory=self._learning_params.pin_memory,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self._learning_params.batch_size,
            shuffle=False,
            num_workers=self._learning_params.num_workers,
            pin_memory=self._learning_params.pin_memory,
        )


# ---------------------------------------------------------------------------
# Build data module from hardcoded lvl1_vqgan config
# ---------------------------------------------------------------------------


CDPAM_SAMPLE_RATE = 22050


def make_cdpam_evaluator(device: str = "cpu") -> _cdpam_lib.CDPAM:
    return _cdpam_lib.CDPAM(dev=device)


def cdpam_distance(
    evaluator: _cdpam_lib.CDPAM,
    pred: torch.Tensor,
    target: torch.Tensor,
    src_sr: int = 44100,
) -> torch.Tensor:
    """Compute mean CDPAM perceptual distance. pred/target: [B, 1, L] at src_sr."""
    resampler = torchaudio.transforms.Resample(src_sr, CDPAM_SAMPLE_RATE)
    pred_r = resampler(pred.cpu().float()).squeeze(1) * 32768.0
    target_r = resampler(target.cpu().float()).squeeze(1) * 32768.0
    with torch.no_grad():
        return evaluator.forward(target_r, pred_r).mean()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Additional perceptual metrics (differentiable, torch-only) — used both as
# eval metrics here and as training losses in train.py. cdpam alone is blind to
# melody/pitch; these add overall spectral fidelity (multi-res STFT) and a
# melody-specific term (chroma = pitch-class energy over time).
# ---------------------------------------------------------------------------


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Magnitude STFT. x: [B,1,L] or [B,L] -> [B, 1+n_fft//2, T]. Differentiable."""
    if x.dim() == 3:
        x = x.squeeze(1)
    # fp32 pinned: cuFFT has no bf16, and a bf16 Trainer precision hands this
    # both a bf16 target batch and a bf16 torch default dtype.
    x = x.float()
    window = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
    )
    return spec.abs()


def multi_res_stft_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    ffts: tuple[int, ...] = (512, 1024, 2048),
) -> torch.Tensor:
    """Spectral-convergence + log-magnitude L1 over several FFT sizes (lower=better).

    The neural-vocoder workhorse: captures harmonic + transient fidelity across
    resolutions. pred/target: [B,1,L] at the same sample rate.
    """
    total = pred.new_zeros(())
    for n_fft in ffts:
        hop = n_fft // 4
        mp = _stft_mag(pred, n_fft, hop)
        mt = _stft_mag(target, n_fft, hop)
        sc = torch.norm(mt - mp) / (torch.norm(mt) + 1e-7)
        lm = F.l1_loss(torch.log(mp + 1e-5), torch.log(mt + 1e-5))
        total = total + sc + lm
    return total / len(ffts)


_CHROMA_FB: dict[tuple[int, int], torch.Tensor] = {}


def _chroma_filterbank(n_fft: int, sr: int, device: torch.device) -> torch.Tensor:
    """Cached [12, 1+n_fft//2] pitch-class filterbank (librosa, fixed weights)."""
    key = (n_fft, sr)
    if key not in _CHROMA_FB:
        import librosa
        import numpy as np

        fb = librosa.filters.chroma(sr=sr, n_fft=n_fft)  # [12, 1+n_fft//2]
        _CHROMA_FB[key] = torch.tensor(np.ascontiguousarray(fb), dtype=torch.float32)
    return _CHROMA_FB[key].to(device)


def chroma_cosine_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    sr: int = 44100,
    n_fft: int = 2048,
) -> torch.Tensor:
    """1 - mean per-frame cosine similarity of chroma (pitch-class) vectors.

    MELODY/harmony metric: robust to timbre/octave, measures whether the right
    notes are present over time. Differentiable. pred/target: [B,1,L].
    """
    hop = n_fft // 4
    mp = _stft_mag(pred, n_fft, hop)  # [B,F,T]
    mt = _stft_mag(target, n_fft, hop)
    fb = _chroma_filterbank(n_fft, sr, pred.device).to(mp.dtype)  # [12,F]
    cp = torch.einsum("cf,bft->bct", fb, mp)  # [B,12,T]
    ct = torch.einsum("cf,bft->bct", fb, mt)
    cp = F.normalize(cp, dim=1)
    ct = F.normalize(ct, dim=1)
    cos = (cp * ct).sum(dim=1)  # [B,T]
    return 1.0 - cos.mean()


def _stft_complex(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Complex STFT. x: [B,1,L] or [B,L] -> [B, 1+n_fft//2, T] complex."""
    if x.dim() == 3:
        x = x.squeeze(1)
    x = x.float()  # fp32 pinned: cuFFT has no bf16
    window = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)
    return torch.stft(
        x, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
    )


def phase_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    ffts: tuple[int, ...] = (512, 1024, 2048),
) -> torch.Tensor:
    """Magnitude-weighted phase incoherence (lower=better), differentiable.

    For each resolution: cos(phase_pred - phase_target) via the normalized complex
    inner product, weighted by target magnitude so silent bins are ignored. This
    isolates PHASE coherence — what melody/pitched content needs and what the
    magnitude-only mrstft/chroma terms are blind to. pred/target: [B,1,L].
    """
    total = pred.new_zeros(())
    for n_fft in ffts:
        hop = n_fft // 4
        sp = _stft_complex(pred, n_fft, hop)
        st = _stft_complex(target, n_fft, hop)
        mp = sp.abs()
        mt = st.abs()
        cos = (sp.real * st.real + sp.imag * st.imag) / (mp * mt + 1e-7)
        w = mt / (mt.mean() + 1e-7)  # emphasize energetic (pitched) bins
        total = total + (w * (1.0 - cos)).mean()
    return total / len(ffts)


# Composite headline weights. cdpam ~0.1-0.3, mrstft ~2.2, chroma ~0.04, phase
# ~0.5-1.5. chroma weighted up (tiny scale); phase added so the metric can see
# the melody/pitch coherence the magnitude terms miss. Heuristic — tune by ear.
COMPOSITE_WEIGHTS = {"cdpam": 1.0, "mrstft": 1.0, "chroma": 2.0, "phase": 1.0}


def perceptual_eval(
    evaluator: _cdpam_lib.CDPAM,
    pred: torch.Tensor,
    target: torch.Tensor,
    src_sr: int = 44100,
) -> dict[str, float]:
    """Panel of perceptual metrics + a composite headline (all lower=better).

    pred/target: [B,1,L] at src_sr. Reports cdpam (timbre similarity), mrstft
    (spectral fidelity), chroma (melody), and a weighted composite so progress
    can't hide melody loss behind a good cdpam.
    """
    with torch.no_grad():
        cd = float(cdpam_distance(evaluator, pred, target, src_sr).item())
        mr = float(multi_res_stft_distance(pred, target).item())
        ch = float(chroma_cosine_distance(pred, target, sr=src_sr).item())
        ph = float(phase_distance(pred, target).item())
    w = COMPOSITE_WEIGHTS
    composite = (
        w["cdpam"] * cd + w["mrstft"] * mr + w["chroma"] * ch + w["phase"] * ph
    )
    return {
        "cdpam": cd,
        "mrstft": mr,
        "chroma": ch,
        "phase": ph,
        "composite": composite,
    }


def build_data_module(learning_params: LearningParameters) -> SplitDatasetModule:
    dataset = MP3SliceDataset(
        data_path=os.path.expanduser("~/.cache/infected_pbm/tracks"),
        sample_rate=44100,
        slice_length=32768,
        device="cpu",
        processed_path=os.path.expanduser("~/.cache/infected_pbm/slices"),
        save_processed=True,
        use_preloaded=False,
        slice_dataset_size=SLICE_DATASET_SIZE,
    )
    return SplitDatasetModule(learning_params=learning_params, dataset=dataset)
