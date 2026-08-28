# Generative Stage — Architecture Notes

Status: planning only. Written 2026-08-25, before the tokenizer is finalized.
Scope: what to build *after* the VQ-GAN tokenizer is good enough, in order to
generate new / combined music from its tokens.

---

## 1. Recommendation summary

1. **Primary path:** autoregressive decoder-only transformer over the RVQ tokens,
   using the **delay pattern** for the depth axis (MusicGen-style).
2. **Run in parallel as a cheap probe:** continuous **latent diffusion transformer
   (DiT) on the pre-quantized `z_e`**, reusing the existing decoder. At small data
   scale this frequently beats token-AR.
3. **Discrete diffusion / diffusion-LM over tokens:** not recommended as the first
   bet. It loses to AR at equal compute in every published audio comparison.
4. **Hierarchical multi-level tokenizers:** the earlier attempt failed for a
   structural reason, not a tuning reason — see §4. Do not retry it as-is.

---

## 2. The depth axis (3–4 RVQ levels)

Four ways to emit `D` codebook entries per frame:

| Scheme | Seq cost | Quality | Notes |
|---|---|---|---|
| Flatten (`t0d0, t0d1, … t1d0, …`) | `D`× longer | Best | Exact factorization, but 4× context and 4× decode steps |
| Parallel (all depths in one step) | 1× | Worst | Assumes depths are conditionally independent given the prefix. They are not — `d1` is literally the residual of `d0` |
| **Delay pattern (MusicGen)** | 1× | Near-flatten | Depth `d` is offset by `d` frames, so `d1` at frame `t` can attend to `d0` at frame `t`. Best cost/quality trade |
| RQ-Transformer depth head | ~1.1× | Best | Large temporal transformer + tiny depth transformer run per frame. Exact factorization, no independence assumption |

**Plan:** start with delay. If `d2`/`d3` sound noisy or the model wastes capacity on
them, swap in a depth head — it is a localized change (add a small transformer over
the `D` axis, keep the temporal trunk).

---

## 3. Two numbers that decide feasibility

These matter more than the architecture choice.

### 3.1 Token rate
`frames/sec × D`. At 50 Hz × 4 depths = 200 tok/s, a 3-minute track is ~36k tokens.
That is a long-context problem before it is a music problem.

- Measure the tokenizer's actual frame rate first.
- If > ~100 tok/s: either add a temporal downsampling stage, or accept short clips
  (10–30 s) for the first generation experiments.
- Context length drives everything downstream (attention cost, batch size, VRAM,
  how much structure the model can even represent).

### 3.2 Data hours
MusicGen trained on ~20k hours. This project has been running single-song and
small-corpus experiments. Below roughly 100 hours, an AR transformer memorizes
rather than generalizes.

This is the real bottleneck. Architecture choice cannot fix it. Options: expand the
corpus, accept a "style model" that memorizes one artist, or lean on the
data-efficiency of continuous latent diffusion (§5).

### 3.3 Codebook health (prerequisite)
Before any generative training, measure per-depth **dead codes** and **entropy**.
Dead codes are wasted logits. Highly skewed per-depth entropy makes the training
loss look great while the model learns nothing — an easy-mode loss that misleads.
This project already has a dead-code counter from the tokenizer work; reuse it.

---

## 4. Why the earlier hierarchical attempt failed

The AudioLM win was **semantic tokens** (w2v-BERT / HuBERT), not multi-scale
*acoustic* tokens. Coarse acoustic tokens are still acoustic — they carry timbre,
not structure. Stacking VQ levels of the same waveform buys almost nothing for
long-range coherence, which is exactly what generation needs.

If a hierarchy is wanted, the top level must be a *different kind of signal*:

- Self-supervised music embeddings (MERT, MuQ) as a conditioning stream, or
- An explicit low-rate structure track (~1–5 Hz): beat, chord, section boundary.

Two-stage then becomes: structure model → acoustic AR conditioned on structure.
That is the hierarchy worth building. Defer it until a flat AR baseline sounds like
*something*.

---

## 5. The continuous alternative (worth one experiment)

The tokenizer's `z_e` is already l2-normalized and there is already a trained
decoder. A DiT operating on continuous `z_e`, skipping VQ entirely for generation:

- No codebook entropy tax, no depth-axis problem, no dead-code holes in the
  distribution.
- This is roughly the Stable Audio recipe.
- At small data scale, continuous latent diffusion has repeatedly outperformed
  token AR.
- Cost is about one extra experiment, since the decoder is reusable as-is.

Run it against the AR baseline on identical clips and A/B by ear.

---

## 6. Generation mode: does infilling matter?

This changes the recommendation:

- **Continuation / from-scratch generation** → AR is fine and is the stronger
  coherence model.
- **Infill, remix, splicing two songs, editing the middle** → AR cannot do it
  (strictly left-to-right). A **masked generative model** (MaskGIT-style:
  SoundStorm, MAGNeT) does infilling natively and decodes 10–20× faster.

"Combined music" as a goal points toward the masked family. Note that discrete
diffusion over tokens and masked generative modelling are essentially the same
family under different names — so option 2 from the original brainstorm re-enters
here, but as an *infilling* tool rather than as a better AR.

---

## 7. Human-preference training (RLHF analogue)

This is a strong fit for this project specifically, because the metric/ear gap is
already documented: GAN checkpoints sound better by ear while scoring worse on
cdpam (`finding_gan_vs_cdpam`). Preference data is the direct fix for exactly that
gap — it replaces a proxy metric with the thing actually being optimized.

### 7.1 Build the ladder in this order

Each rung is useful on its own; do not skip to the last one.

1. **A/B harness.** Blind, forced-choice, randomized left/right, keyboard-driven,
   logs to a file. Allow a "tie / can't tell" option and log it. Clip length and the
   question asked are design decisions — see §7.2. This rung alone is worth building
   now: it makes every current checkpoint comparison rigorous instead of
   impressionistic.
2. **Checkpoint ELO.** Feed A/B results into an ELO or Bradley-Terry ranking over
   checkpoints. This *replaces the composite metric* as the arbiter of "is this run
   better". Directly applicable to the tokenizer work happening now, before any
   generative model exists.
3. **Reward model (RM).** Train a scorer on the pairs. Initialize from the
   tokenizer's encoder — it already has good audio features, so the RM is a small
   head, not a from-scratch model. Two heads, not one — see §7.2.
4. **Best-of-N reranking.** Sample N candidates from the generator, keep the RM's
   favorite. No training instability, no risk of degrading the generator, immediate
   audible gain. This is the highest value-per-risk rung, the RM doubles as an
   automatic eval metric, and — critically — it leaves the base policy untouched, so
   sample diversity survives (§7.3).
5. **DPO on the generator.** Only after the RM shows it agrees with the ear on
   held-out pairs.

### 7.2 Two axes, two tiers — the central harness decision

**Do not ask a single "which is better?" question.** One question collapses two
independent axes into whichever is easier to perceive, and in short clips that is
always fidelity. The result is a reward model that only knows "clean", which is the
main route to boring output (§7.3).

Split the collection into two tiers:

| Tier | Clip length | Question | Volume | Trains |
|---|---|---|---|---|
| Bulk | ~10 s | "which sounds cleaner?" | High | Fidelity head |
| Structure | ~30–60 s | "which is more interesting?" | Low | Interest head |

Rationale:

- **Short clips are structurally biased toward fidelity.** Within 10 seconds, only
  cleanliness is observable — artifacts, roughness, smearing. Interestingness needs
  time to reveal itself: development, contrast, a section change. A harness built
  only on 10-second clips cannot see the axis that matters most for generation.
- **Long clips are expensive.** Time per pair explodes, so the same listening budget
  buys far fewer pairs. There is also strong recency bias — comparing two 60-second
  clips, the rater mostly compares the last 10 seconds they heard. And 60 s at
  200 tok/s is ~12k tokens of generation per candidate (§3.1). Hence: low volume,
  reserved for the question that actually needs the duration.
- **Two heads give an explicit knob.** Best-of-N reranks on a weighted combination
  of the two scores. The clean/interesting trade-off becomes a dial to turn, rather
  than something to hope emerges from a single scalar.

For the structure tier, consider a same-clip design where both candidates share a
prefix and diverge — it removes "which intro did I like" from the comparison and
isolates the development.

### 7.3 Why preference training drifts toward boring

This is a property of the objective, not a flaw in the rater. It happens even with a
perfect, honest, self-aware annotator. Two independent mechanisms:

**Variance penalty.** Interesting output is high-variance: sometimes brilliant,
often broken. A model that is 20% brilliant / 80% broken loses more pairwise
comparisons than one that is 100% clean-but-dull. The argmax of average win rate is
not the mode of "great music". The rater never picked "boring" — they picked "not
broken", eight times out of ten, and the optimizer read that as a preference for the
low-variance policy.

**Diversity is invisible to A/B by construction.** Variety is a property *across*
samples, and every pairwise comparison sees exactly two. Nothing in the preference
objective constrains it, so DPO's log-probability push is free to narrow the
sampling distribution. Every individual sample still wins its pair while the model
quietly becomes more confident and less varied.

Countermeasures:

- Prefer best-of-N (rung 4) over DPO (rung 5) for as long as it keeps paying —
  reranking cannot collapse a policy it does not update.
- Ask the "interesting" question explicitly (§7.2) so the axis is at least
  represented in the data.
- **Measure diversity outside the preference loop:** sample N candidates from the
  same conditioning, compute mean pairwise distance in encoder feature space, and
  track it across DPO steps. Gate DPO on that number not dropping. Pairwise data
  will never surface this on its own.
- Track sample entropy and codebook usage during DPO alongside the diversity metric.

### 7.4 DPO specifics

- DPO, not PPO. PPO needs far more data and far more babysitting; DPO trains
  directly on pairs and is stable at this scale.
- Pairs must share conditioning — same prompt/seed context, different sample.
  Otherwise the model learns the prompt distribution, not quality.
- Sequence-level log-probability over the token sequence; `beta` around 0.1.
- Watch for the degeneration described in §7.3, and keep listening to unconditional
  samples throughout, not just to reranked ones.
- For the continuous DiT path the analogue is **Diffusion-DPO** — same idea, works
  on the denoising objective.

### 7.5 Data volume, realistically

- ~200–500 pairs: enough for DPO signal on a small model.
- ~1–3k pairs: minimum for a reward model that generalizes.
- Those numbers are per head. The structure tier will always be the smaller set;
  plan for the interest head to be the weaker of the two and weight it accordingly.
- With a single annotator, this is many hours of listening. Budget for it, and treat
  the A/B harness's ergonomics as a first-class concern — a 3-second-per-pair UI
  versus a 15-second-per-pair UI is the difference between feasible and not.

### 7.6 Annotator noise (single-rater problem)

One rater is cheap but drifts, both within a session (fatigue) and across weeks
(taste shift). The RM ends up trained on a mixture of past selves. Mitigate:

- Inject **repeat pairs** (the same comparison shown again later) at ~10% rate.
- Measure self-agreement, per tier. If it is below ~80%, label noise dominates and
  no amount of RM capacity will help — simplify the comparison (shorter clips, more
  distinct candidates) before collecting more data. Expect the structure tier to be
  noisier than the bulk tier; that is normal, not a reason to abandon it.
- Include a few **anchor pairs** with an obvious right answer to detect fatigue
  sessions, and discard sessions that fail them.

### 7.7 Reward hacking — the specific trap here

A reward model trained on raw audio pairs will learn **loudness and brightness**
proxies, because louder and brighter reliably wins blind A/B tests. The generator
then learns to be a compressor.

This is preattentive, not a matter of taste — it survives training and
self-awareness, which is why every mastering comparison tool auto level-matches. The
rater cannot un-hear +2 LU; they can only avoid being asked the question.

Mitigation, non-optional: **LUFS-normalize every clip** before it is played in the
A/B harness and before it reaches the RM. The cost is near zero, so this is
insurance rather than a judgment about anyone's ear.

The same failure mode extends past gain: spectral brightness, stereo width, and
transient sharpness all win a 10-second A/B and all fatigue over an album length.
Periodically sanity-check the RM by scoring pairs where the "worse" clip is simply
the better one, quieter.

### 7.8 Do not bootstrap the RM from cdpam
Tempting, but self-defeating: cdpam disagreeing with the ear is the entire reason
for building this. Pre-labelling with cdpam would bake the disagreement into the RM.
Bootstrap from the *encoder features* instead, not from the metric.

## 8. Concrete build order

1. Freeze the tokenizer. Precompute tokens (and `z_e`) for the whole corpus to disk.
   Generative iterations then take minutes, not hours.
2. Report codebook health: dead codes and entropy per depth (§3.3).
3. Build the A/B harness (§7.1 rung 1, PySide6 — §9), two-tier and two-question from the start
   (§7.2) with LUFS normalization on playback (§7.7) — retrofitting either means
   discarding the pairs already collected. It pays off immediately on tokenizer
   work, before any generative model exists; the bulk tier is usable there as-is.
4. AR baseline: small decoder-only transformer, delay pattern, 10–20 s clips. Target
   is "coherent at all", not "good".
5. Probe: latent DiT on continuous `z_e`, same clips, same decoder.
6. Ear A/B between 4 and 5. Pick a lane.
7. Add the conditioning stream for long-range structure (§4).
8. Preference ladder rungs 2→5 (§7.1). Stay on rung 4 (best-of-N) as long as it
   keeps paying — rung 5 is where diversity collapse lives (§7.3).

Ear A/B decides, not loss curves — the lesson already learned from GAN vs cdpam.

---

## 9. UI stack — PySide6

`pyside6>=6.11.2` added to `pyproject.toml` on 2026-08-25 (project is on Python
>=3.13). Two intended uses, in order:

1. **Near term: the A/B harness (§7.1 rung 1).** This is the thing that has to exist
   before any preference data can be collected, and its ergonomics decide whether
   collection is feasible at all (§7.5).
2. **Later: a front-facing app.** Generation UI over the finished tokenizer +
   generative model. Same toolkit, so harness widgets and audio plumbing carry over
   rather than being thrown away.

### 9.1 Audio playback

QtMultimedia covers it without extra deps:

- `QMediaPlayer` + `QAudioOutput` — file playback, simplest path.
- `QAudioSink` — push raw PCM from a numpy buffer; use this once clips are being
  generated in-process and never hit disk.
- `QSoundEffect` — low-latency but WAV-only; fine for UI feedback, not for rating
  material.

Still missing from `pyproject.toml` if going the numpy route: `soundfile` (I/O) and
`pyloudnorm` (§7.7 normalization). Add when the harness is actually built, not
before.

### 9.2 Harness design notes

These follow from §7 and are cheap to build in now, expensive to retrofit:

- **Preload both clips into memory before the pair is shown.** Decode + LUFS-match
  up front. Any load delay at play time is the single biggest threat to the
  3-seconds-per-pair target.
- **Instant A/B toggle at a shared playhead.** Switch between candidate A and B
  mid-playback without restarting, the way mastering comparison tools do it. Far
  more discriminating than sequential playback, and it is the only practical way to
  judge the bulk (fidelity) tier.
- Sequential playback is still needed for the structure tier (§7.2) — judging
  development over 30–60 s means hearing each candidate whole.
- **Keyboard-only rating.** A/B/tie on three keys, no mouse, auto-advance to the
  next pair.
- **Blind and randomized**: which candidate is A must be randomized per pair, and
  checkpoint identity must never be visible during rating.
- **Append-only log**, one row per judgement: pair id, tier, question asked, both
  candidate ids, choice, response time, session id. Response time is worth logging —
  it flags the pairs the rater found hard, and hard pairs are the informative ones.
- Repeat pairs and anchor pairs (§7.6) get injected by the pair sampler, not by the
  UI; the UI must not be able to tell them apart from normal pairs.
