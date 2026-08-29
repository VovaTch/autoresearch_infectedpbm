"""
Blind A/B preference harness over AR-generated music (PLAN section 7.1 rung 1).

Layered MVVM so the domain and view-model tiers stay testable without a display,
a GPU or an audio device:

  model/      plain Python + numpy. No Qt, no torch.
  worker/     the generation service. Owns the GPU, runs in its own process.
  viewmodel/  QObject + signals. No widgets.
  view/       QWidget only. No business logic.
"""
