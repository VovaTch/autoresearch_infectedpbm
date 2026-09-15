"""
Slice synthesizer: drive the AR model by hand.

The rating harness (ab_harness) draws conditioning at random and hides it,
because its job is blind preference collection. This app inverts that --
conditioning IS the interface: pick tracks, pick or randomize style embeddings,
optionally prime with real audio, and hear every combination.

Layering matches ab_harness, and for the same reason:

  model/      plain Python and numpy. No Qt, no torch. Testable anywhere.
  worker/     owns the GPU, the checkpoint and the ONNX graphs. Runs in a child
              process so the UI never imports torch.
  viewmodel/  QObject and signals, no widgets.
  view/       widgets, no logic.

Heavy lifting is borrowed from ab_harness rather than reimplemented: the sampler
(worker.generator), the decoder (worker.decoder), checkpoint loading
(worker.loading), loudness and envelopes (model.audio), and the playback
transport (viewmodel.player_vm). Nothing about the rating session -- the bank,
the tiers, the judgement log -- is shared.
"""
