"""
Keeps pairs ready ahead of the rater.

PLAN section 7.5 makes the ergonomics a first-class concern: a 3-second-per-pair
UI versus a 15-second one is the difference between the data being collectable
and not. Everything here exists to stop the rater ever waiting on a GPU.

Two lanes feed the queue the rater sees:

  * Fresh generations, for the bulk tier. A 10 s clip is a few seconds of
    sampling, so at the default prefetch depth the queue stays ahead of anyone
    rating at a human pace.
  * Already-banked tokens, which only need decoding. Repeats and anchors are
    always in this lane, and so is the structure tier by default.

Behind them runs a third, non-serving lane: the structure backfill. A 90 s pair
is around two minutes of sampling against roughly ten seconds of rating, so the
structure tier can never be produced on demand -- the GPU is some three times
too slow for the configured share. What it can do is run in the background
whenever the rating queue is already full, banking one structure pair at a time.
Those clips are then in the bank for every later draw and every later session,
so the tier fills in on its own instead of requiring a manual bake.

Until something is banked, structure draws are declined and redrawn as bulk.
That is counted in `skipped_structure` and surfaced in the status bar: silently
discarding a fifth of all draws is how a rater ends up doing thirty comparisons
without meeting the tier they configured.

The bank is re-indexed as the queue is topped up, because the worker process
owns every write and the UI's copy is otherwise a snapshot taken at startup.
Without that rescan the backfill's output only became servable in the *next*
session, so a first session could bank twenty structure pairs and show none.

The queue is also addressable: worklist() publishes what is ready and what is
coming, take() serves a chosen entry, and request() queues a tier on demand.
A random tier mix is a coin flip per pair, not a schedule, so a 20% share can
easily show up once in thirteen pairs -- picking from a list is the fix.

A spec whose clips fail to produce is dropped rather than retried. The rater
sees one fewer pair; they never see a stall.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from ab_harness.model.pair_sampler import PairSampler
from ab_harness.model.protocols import ClipProducer, TokenStore
from ab_harness.model.types import Clip, ClipSpec, Pair, PairSpec, Tier

MAX_REDRAWS = 8
REFRESH_INTERVAL_S = 2.0


@dataclass(frozen=True)
class WorkItem:
    """
    One entry of the rater's worklist.

    Carries nothing that identifies a candidate: the tier and the length are
    shared by both sides, so a list row cannot leak which clip is which.

    Args:
      pair_id (str): the comparison's id, used to select it.
      tier (Tier): rating tier.
      seconds (float): clip length, so 10 s and 90 s rows are told apart.
      pair (Pair | None): the decoded comparison, or None while it is still
        being produced.
    """

    pair_id: str
    tier: Tier
    seconds: float
    pair: Pair | None = None

    @property
    def ready(self) -> bool:
        """
        Returns:
          bool: True when this item can be rated now.
        """
        return self.pair is not None


class PairPipeline:
    """
    Prefetching PairSource over a sampler and a producer.

    Args:
      sampler (PairSampler): draws what to compare.
      producer (ClipProducer): turns specs into audio.
      store (TokenStore): the bank, consulted to tell fast decodes from
        generations.
      depth (int): pairs kept ready or in flight.
      structure_live (bool): allow structure-tier pairs to be generated on
        demand rather than only served from the bank. Off by default: it stalls
        the queue for two minutes. The backfill lane is the better route.
      structure_backfill (int): structure pairs generated in the background at
        once, submitted only once the rating queue is full. 0 disables it.
      rated_items (Iterable[str] | None): item ids already judged, from earlier
        sessions. Banked material the rater has never heard is served first, so
        a bank that keeps growing does not keep replaying its oldest pairs.
      min_fill (float): a pair whose quieter side is less full of audio than
        this never reaches the worklist; it is dropped and redrawn. 0 disables
        the gate. The model has a low-energy attractor -- measured over 200
        clips, the 90 s tier averages 30% near-silent against 0% for the real
        tokens of the same spans -- and a dead clip costs a listen while
        teaching a preference model only that energy wins, which is the
        section 7.7 reward-hacking trap.
    """

    def __init__(
        self,
        sampler: PairSampler,
        producer: ClipProducer,
        store: TokenStore,
        depth: int = 3,
        structure_live: bool = False,
        structure_backfill: int = 1,
        rated_items: Iterable[str] | None = None,
        min_fill: float = 0.0,
    ) -> None:
        self.sampler = sampler
        self.producer = producer
        self.store = store
        self.depth = max(1, depth)
        self.structure_live = structure_live
        self.structure_backfill = max(0, structure_backfill)
        self.min_fill = max(0.0, min_fill)
        self._ready: deque[Pair] = deque()
        self._inflight: dict[str, PairSpec] = {}
        self._clips: dict[str, Clip] = {}
        self._backfill: dict[str, PairSpec] = {}
        self.rated_items: set[str] = set(rated_items or ())
        self.dropped = 0
        self.skipped_structure = 0
        self.rejected_quiet = 0
        self._last_refresh = 0.0

    # -- queue state ---------------------------------------------------------

    @property
    def ready(self) -> int:
        """
        Returns:
          int: pairs decoded and waiting to be shown.
        """
        return len(self._ready)

    @property
    def inflight(self) -> int:
        """
        Returns:
          int: pairs submitted but not yet complete.
        """
        return len(self._inflight)

    @property
    def backfilling(self) -> int:
        """
        Returns:
          int: structure pairs being generated in the background.
        """
        return len(self._backfill)

    def banked_structure_pairs(self) -> int:
        """
        Returns:
          int: structure groups in the bank holding at least two clips, i.e. how
            many structure comparisons can currently be served.
        """
        self._refresh_store()
        groups: dict[str, int] = {}
        for spec in self.store.specs(Tier.STRUCTURE):
            groups[spec.group_id] = groups.get(spec.group_id, 0) + 1
        return sum(1 for count in groups.values() if count >= 2)

    # -- worklist ------------------------------------------------------------

    def worklist(self) -> list[WorkItem]:
        """
        The queue as the rater sees it: what can be rated now, and what is coming.

        Ready pairs come first, in the order they would be served. The backfill
        lane is excluded -- it exists to fill the bank, not to be rated.

        Returns:
          list[WorkItem]: ready pairs then in-flight ones.
        """
        fps = self.sampler.cfg.fps
        items = [
            WorkItem(
                pair_id=pair.spec.pair_id,
                tier=pair.spec.tier,
                seconds=pair.spec.left.n_frames / fps,
                pair=pair,
            )
            for pair in self._ready
        ]
        items += [
            WorkItem(
                pair_id=spec.pair_id,
                tier=spec.tier,
                seconds=spec.left.n_frames / fps,
                pair=None,
            )
            for spec in self._inflight.values()
        ]
        return items

    def take(self, pair_id: str) -> Pair | None:
        """
        Serve one specific ready pair instead of the head of the queue.

        This is what makes the worklist more than a display: with a random draw
        the rater met the 90 s tier once in thirteen pairs, because the tier mix
        is a coin flip per pair rather than a schedule.

        Args:
          pair_id (str): the comparison to show.

        Returns:
          Pair | None: the pair, or None if it is not ready.
        """
        for index, pair in enumerate(self._ready):
            if pair.spec.pair_id != pair_id:
                continue
            del self._ready[index]
            self._release(pair)
            self._topup()
            return pair
        return None

    def give_back(self, pair: Pair) -> None:
        """
        Return a shown-but-unrated pair to the head of the queue.

        A worklist the rater can jump around in has to be non-destructive:
        selecting a second entry while the first is on screen would otherwise
        throw the first away unjudged.

        Args:
          pair (Pair): the comparison leaving the screen without a decision.
        """
        if any(ready.spec.pair_id == pair.spec.pair_id for ready in self._ready):
            return
        self._ready.appendleft(pair)

    def request(self, tier: Tier, count: int = 1) -> int:
        """
        Queue comparisons of one tier on demand.

        Bypasses both the tier mix and the structure decline: an explicit ask is
        the rater deciding what to spend the GPU on, which is the only way to
        get 90 s pairs at a rate the configured share never reaches.

        Structure requests are filled from the bank first when unrated material
        is there, since decoding is seconds against two minutes of sampling.

        Args:
          tier (Tier): which tier to produce.
          count (int): how many pairs.

        Returns:
          int: pairs queued.
        """
        queued = 0
        for _ in range(max(0, count)):
            spec: PairSpec | None = None
            if tier is Tier.STRUCTURE:
                self._refresh_store()
                spec = self._banked_structure_spec(fresh_only=True)
            if spec is None:
                spec = self.sampler.next_spec(tier=tier)
            self._inflight[spec.pair_id] = spec
            self.producer.submit([spec.left, spec.right])
            queued += 1
        return queued

    def mark_rated(self, spec: PairSpec) -> None:
        """
        Record that a comparison has been judged.

        Args:
          spec (PairSpec): the comparison just rated. Its clips stop counting as
            unheard, so banked draws move on to material the rater has not met.
        """
        self.rated_items.update((spec.left.item_id, spec.right.item_id))

    # -- drawing -------------------------------------------------------------

    def _refresh_store(self) -> None:
        """
        Re-index the bank, at most every REFRESH_INTERVAL_S seconds.

        The backfill lane banks structure clips from the worker process, so the
        UI-side store is stale by construction. Without this the tier a rater
        configured stays invisible for the whole session no matter how much of
        it the GPU produced.
        """
        now = time.monotonic()
        if now - self._last_refresh < REFRESH_INTERVAL_S:
            return
        self._last_refresh = now
        self.store.refresh()

    def _is_banked(self, spec: PairSpec) -> bool:
        """
        Args:
          spec (PairSpec): a candidate comparison.

        Returns:
          bool: True when both clips only need decoding.
        """
        return self.store.has(spec.left.item_id) and self.store.has(spec.right.item_id)

    def _draw(self) -> PairSpec | None:
        """
        Draw a comparison this pipeline is willing to produce.

        Returns:
          PairSpec | None: a spec, or None when repeated draws all landed on the
            structure tier with nothing banked to serve.
        """
        self._refresh_store()
        for _ in range(MAX_REDRAWS):
            spec = self.sampler.next_spec()
            if (
                spec.tier is not Tier.STRUCTURE
                or self.structure_live
                or self._is_banked(spec)
            ):
                return spec
            # A freshly drawn structure spec is a fresh random recipe, so it is
            # never what the backfill happened to bank. Substitute a banked
            # comparison instead, or the tier could only ever appear by
            # coincidence.
            banked = self._banked_structure_spec()
            if banked is not None:
                return banked
            self.skipped_structure += 1
        return None

    def _banked_structure_spec(self, fresh_only: bool = False) -> PairSpec | None:
        """
        Assemble a structure comparison out of clips already in the bank.

        Groups holding nothing the rater has judged are preferred, so a bank
        that keeps growing is worked through rather than replayed.

        Args:
          fresh_only (bool): return None instead of falling back to a group
            whose clips have already been rated.

        Returns:
          PairSpec | None: a blinded pair from a banked group holding at least
            two clips, or None when there is no such group.
        """
        groups: dict[str, list[ClipSpec]] = {}
        for spec in self.store.specs(Tier.STRUCTURE):
            groups.setdefault(spec.group_id, []).append(spec)
        usable = [
            [
                m
                for m in members
                if self.store.has(m.item_id) and not self._known_quiet(m.item_id)
            ]
            for members in groups.values()
        ]
        usable = [members for members in usable if len(members) >= 2]
        fresh = [
            members
            for members in usable
            if not any(m.item_id in self.rated_items for m in members)
        ]
        pool = fresh or ([] if fresh_only else usable)
        if not pool:
            return None
        members = self.sampler.rng.choice(pool)
        left, right = self.sampler.rng.sample(members, 2)
        spec = self.sampler.blind(Tier.STRUCTURE, left, right)
        self.sampler.register(spec)
        return spec

    def _topup(self) -> None:
        """Submit new pairs until depth pairs are ready or in flight."""
        starved = False
        while self.ready + self.inflight < self.depth:
            spec = self._draw()
            if spec is None:
                starved = True
                break
            self._inflight[spec.pair_id] = spec
            # Both sides go out every time: the producer reuses banked tokens
            # itself, so there is no second code path to keep in step.
            self.producer.submit([spec.left, spec.right])
        self._topup_backfill(starved)

    def _topup_backfill(self, starved: bool) -> None:
        """
        Keep the background structure lane busy without stalling the rater.

        The worker is FIFO, so a structure batch costs roughly two minutes
        during which no bulk clip comes back. It is therefore only started with
        a full queue of *ready* pairs to rate through -- in-flight work is no
        protection, since it is queued behind the structure batch.

        The exception is a starved main lane with nothing in flight: then the
        rater is waiting either way, and refusing to backfill would deadlock a
        bank that has no structure material to bootstrap from.

        Args:
          starved (bool): the main lane could not find anything to submit.
        """
        if not self.structure_backfill:
            return
        idle = starved and self.inflight == 0
        if not idle and self.ready < self.depth:
            return
        while len(self._backfill) < self.structure_backfill:
            spec = self.sampler.next_spec(tier=Tier.STRUCTURE)
            if self._is_banked(spec):
                return
            self._backfill[spec.pair_id] = spec
            self.producer.submit([spec.left, spec.right])

    # -- collection ----------------------------------------------------------

    def pump(self, timeout: float = 0.0) -> None:
        """
        Collect finished clips, assemble complete pairs, and top the queue up.

        Collection runs again after the top-up because a synchronous producer
        finishes inside submit(); without the second pass the first pair of a
        session would always come back empty.

        Args:
          timeout (float): seconds to wait for the first clip.
        """
        self._collect(timeout)
        self._assemble()
        self._topup()
        self._collect(0.0)
        self._assemble()

    def _collect(self, timeout: float) -> None:
        """
        Args:
          timeout (float): seconds to wait for the first clip.
        """
        for clip in self.producer.poll(timeout):
            self._clips[clip.spec.item_id] = clip

    def _too_quiet(self, pair: Pair) -> bool:
        """
        Args:
          pair (Pair): a freshly assembled comparison.

        Returns:
          bool: True when either side is emptier than min_fill. References are
            exempt: real tokens are never dead air, and an anchor has to keep
            its known-correct side whatever it measures.
        """
        if not self.min_fill:
            return False
        for clip in (pair.left, pair.right):
            if clip.spec.is_reference or clip.fill < 0.0:
                continue
            if clip.fill < self.min_fill:
                return True
        return False

    def _known_quiet(self, item_id: str) -> bool:
        """
        Args:
          item_id (str): a clip id.

        Returns:
          bool: True when the bank already measured this clip below min_fill,
            so it can be skipped without decoding it again.
        """
        if not self.min_fill:
            return False
        recorded = getattr(self.store, "fill", lambda _: None)(item_id)
        return recorded is not None and recorded < self.min_fill

    def _assemble(self) -> None:
        """Move every in-flight pair whose clips have both arrived to ready."""
        for pair_id, spec in list(self._inflight.items()):
            left = self._clips.get(spec.left.item_id)
            right = self._clips.get(spec.right.item_id)
            if left is None or right is None:
                continue
            del self._inflight[pair_id]
            pair = Pair(spec=spec, left=left, right=right)
            if self._too_quiet(pair):
                # Dropped rather than shown. The tokens stay banked with their
                # measured fill, so the same dead clip is never drawn again.
                self.rejected_quiet += 1
                self._release(pair)
                continue
            self._ready.append(pair)
        self._retire_backfill()

    def _retire_backfill(self) -> None:
        """
        Forget finished backfill pairs and drop their audio.

        The backfill exists to put tokens in the bank, not to serve anyone, so
        its decoded audio is dead weight the moment both clips have landed --
        and at 7.9 MB per 90 s clip it is weight worth dropping promptly.
        """
        for pair_id, spec in list(self._backfill.items()):
            items = (spec.left.item_id, spec.right.item_id)
            if not all(item in self._clips for item in items):
                continue
            del self._backfill[pair_id]
            for item in items:
                self._clips.pop(item, None)

    def drop_stalled(self, item_ids: Sequence[str]) -> None:
        """
        Abandon pairs waiting on clips the producer failed to make.

        Args:
          item_ids (Sequence[str]): clip ids that will never arrive.
        """
        lost = set(item_ids)
        for pair_id, spec in list(self._inflight.items()):
            if {spec.left.item_id, spec.right.item_id} & lost:
                del self._inflight[pair_id]
                self.dropped += 1
        for pair_id, spec in list(self._backfill.items()):
            if {spec.left.item_id, spec.right.item_id} & lost:
                del self._backfill[pair_id]

    # -- PairSource ----------------------------------------------------------

    def next_pair(self) -> Pair | None:
        """
        Hand over the next comparison.

        Returns:
          Pair | None: a ready pair, or None while the queue is still filling.
            Callers poll again rather than block, so the UI stays responsive.
        """
        self.pump()
        if not self._ready:
            return None
        pair = self._ready.popleft()
        self._release(pair)
        self._topup()
        return pair

    def _release(self, pair: Pair) -> None:
        """
        Forget the audio of a pair that has left the queue.

        Args:
          pair (Pair): the pair being handed out. Its clips stay alive through
            the returned object; dropping them here keeps memory flat across a
            long session, since a 90 s clip is 7.9 MB.
        """
        pending = list(self._inflight.values()) + list(self._backfill.values())
        for item_id in (pair.spec.left.item_id, pair.spec.right.item_id):
            if not any(item_id in (s.left.item_id, s.right.item_id) for s in pending):
                self._clips.pop(item_id, None)
