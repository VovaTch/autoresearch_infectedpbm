# Finding: z_e BatchNorm vs l2-norm (2026-07-01 → 07-02)

## Context
VQ-GAN music reconstruction (Infected Mushroom tracks). Goal: most faithful
reconstruction to the **human ear**. Metrics: cdpam (lower=better, primary),
mrstft (lower=better). Encoder emits `z_e` (continuous latent) before residual
VQ. Encoder tail normalizes `z_e` to pin its scale — else align/commit MSE
(both computed vs `z_e`) run away under high LR.

Baked baseline (see memory `finding_ze_norm_fix`): **l2** per-token unit-norm,
`F.normalize(z_e, dim=1)` in `TemporalEncoder.forward`. Overturned earlier grms.
Best l2 full-data runs (long3/long4): composite ~1.97, **cdpam ~0.102**, mrstft ~1.20.

## Hypothesis tested
l2 per-token norm forces every token onto unit sphere → z_e scale gets small,
erases magnitude info. Try `nn.BatchNorm1d(token_dim)` at encoder tail instead:
per-channel normalize across (B,T), keeps learnable affine (scale+shift) and
lets magnitude vary per token.

## Run
- Swap: `return F.normalize(z_e, dim=1)` → `self._bn_end = nn.BatchNorm1d(token_dim)`; `return self._bn_end(z_e)`
- From scratch, 2-GPU DDP, full dataset, one-cycle LR peak 1.4142e-4, ~14.5h wall.
- 149M params, token_dim=1024, hidden=1024, num_rq=3, num_tokens=2048, time_downsample=1.

## Result — REGRESSED
- **alignment_loss CLIMBED to 8.64** (l2 keeps align flat/low). Core failure.
- Test cdpam **0.125** vs l2 baseline **~0.102** — worse.
- Per-clip render (EMA weights), tracks Nevermind + Desert_Storm:

  | clip | cdpam | mrstft |
  |------|-------|--------|
  | Nevermind p25 | 0.129 | 0.234 |
  | Nevermind p50 | 0.105 | 0.260 |
  | Nevermind p75 | 0.168 | 0.286 |
  | Desert_Storm p25 | 0.113 | 0.291 |
  | Desert_Storm p50 | 0.122 | 0.253 |
  | Desert_Storm p75 | 0.050 | 0.431 |

  mean cdpam ~0.121.

## Why it likely failed
- BatchNorm's per-channel stats + learnable affine let z_e magnitude drift →
  align/commit targets move → alignment can't converge (climbed to 8.64).
- l2's fixed unit-norm matches codebook init scale (~1); batchnorm has no such
  anchor to the codebook.
- Extra caveat (not root cause but real): BatchNorm1d NOT synced across DDP ranks
  (no SyncBatchNorm) → per-GPU running stats; and inference uses running mean/var
  (eval) vs l2's deterministic per-sample norm → train/inference mismatch.

## Decision
**Reverted to l2** (`F.normalize(z_e, dim=1)`, `_bn_end` removed). l2 remains the
baked default. BatchNorm at encoder tail = dead end for this arch.

## State after revert (current working tree)
- `train.py`: l2 restored, batchnorm removed.
- `render_samples.py`: TRACKS changed to `["deeply_disturbed", "Cookie_From_Space"]`.
- `config.yaml`: minutes still 870, devices 2, single_song null, checkpoint null.
- Checkpoints from batchnorm run still in `saved/` — stale, ignore/overwrite.

## Open levers (from memory `finding_multigpu_codebook_collapse`)
- TIME+LR tapped out (composite flatlined ~1.97 after ~101h). Low-LR fine-tune = wash.
- **BIGGER MODEL** is the main remaining lever. Pipeline wired: `--hidden` (conv
  width) + `--init-from` partial-loader. hidden=1024 (145M, 3.5x) fits VRAM.
- z_e normalization now explored: grms → l2 (winner) → batchnorm (lost). l2 stays.
