"""
On-disk store of generated token streams.

Only tokens are persisted. Audio is decoded into RAM on demand and thrown away,
because tokens are what the reward model and DPO actually consume -- the wavs
were only ever playback material. A 10 s item is 1723x3 int16 = 10 kB and a 90 s
item is 93 kB, so a whole bank is smaller than one of the wav renders it
replaces.

The original tracks and their real tokens are not copied in either. They already
live in ~/.cache/infected_pbm/tracks/ and ~/.cache/infected_pbm/tokens_*/, and an
item's (track_idx, start_frame, n_frames) is the pointer into them. The manifest
records the token cache tag so a bank can never be silently paired with tokens
from a different tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ab_harness.model.types import ClipSpec, Tier

BANK_VERSION = 1


class TokenizerMismatch(RuntimeError):
    """Raised when a bank and a token cache come from different tokenizers."""


class ClipBank:
    """
    Token + metadata store rooted at one directory.

    Layout::

        manifest.json
        items/<item_id>.npy    (T, R) int16 codes
        items/<item_id>.json   the ClipSpec
        sessions/<session>.jsonl
        judgements.tsv

    Args:
      root (Path): bank directory; created if absent.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser()
        self.items_dir = self.root / "items"
        self.sessions_dir = self.root / "sessions"
        self.items_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, Any] | None = None
        self._fills: dict[str, float] | None = None
        self._specs: dict[str, ClipSpec] = {}
        self._order: list[str] = []
        self._load_specs()

    # -- manifest ------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        """
        Returns:
          Path: the manifest file, which may not exist yet.
        """
        return self.root / "manifest.json"

    @property
    def judgements_tsv(self) -> Path:
        """
        Returns:
          Path: the flat mirror of every session log.
        """
        return self.root / "judgements.tsv"

    def manifest(self) -> dict[str, Any]:
        """
        Returns:
          dict[str, Any]: the manifest, empty if the bank was never initialised.
        """
        if self._manifest is None:
            if self.manifest_path.exists():
                self._manifest = json.loads(self.manifest_path.read_text())
            else:
                self._manifest = {}
        return self._manifest

    def init_manifest(
        self, token_cache_tag: str, tokenizer_meta: dict[str, Any], checkpoint: str
    ) -> None:
        """
        Write the manifest, or verify an existing one still matches.

        Args:
          token_cache_tag (str): directory name of the token cache in use.
          tokenizer_meta (dict[str, Any]): contents of tokenizer_meta.json.
          checkpoint (str): AR checkpoint tag producing new items.

        Raises:
          TokenizerMismatch: if the bank was built against a different tokenizer.
        """
        current = self.manifest()
        if current:
            self.require_tokenizer(token_cache_tag)
            seen = set(current.get("checkpoints", []))
            if checkpoint and checkpoint not in seen:
                current["checkpoints"] = sorted(seen | {checkpoint})
                self._write_manifest(current)
            return
        self._write_manifest(
            {
                "version": BANK_VERSION,
                "token_cache_tag": token_cache_tag,
                "tokenizer_meta": tokenizer_meta,
                "checkpoints": [checkpoint] if checkpoint else [],
            }
        )

    def require_tokenizer(self, token_cache_tag: str) -> None:
        """
        Args:
          token_cache_tag (str): the tag the caller intends to use.

        Raises:
          TokenizerMismatch: if it differs from the one the bank was built with.
        """
        stored = self.manifest().get("token_cache_tag")
        if stored and stored != token_cache_tag:
            raise TokenizerMismatch(
                f"bank was built against {stored!r}, not {token_cache_tag!r}; "
                "tokens from different tokenizers are not comparable"
            )

    def _write_manifest(self, data: dict[str, Any]) -> None:
        """
        Args:
          data (dict[str, Any]): manifest contents to persist.
        """
        self.manifest_path.write_text(json.dumps(data, indent=2))
        self._manifest = data

    # -- items ---------------------------------------------------------------

    def _load_specs(self) -> None:
        """Index every spec already on disk, sorted by id for determinism."""
        self.refresh()

    def refresh(self) -> int:
        """
        Index items written since this bank object was built.

        The worker process owns every write, so a UI-side bank is a snapshot
        taken at startup and goes stale the moment the backfill lane banks
        anything. Without rescanning, a structure pair generated during a
        session can never be served in that session -- the pipeline keeps
        declining structure draws against an index that predates them.

        A torn write is skipped rather than fatal; add() lands the tokens
        before the spec, so the next scan picks it up.

        Returns:
          int: number of newly indexed specs.
        """
        added = 0
        for path in sorted(self.items_dir.glob("*.json")):
            if path.stem in self._specs:
                continue
            try:
                spec = ClipSpec.from_json(json.loads(path.read_text()))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            self._specs[spec.item_id] = spec
            self._order.append(spec.item_id)
            added += 1
        return added

    def add(self, spec: ClipSpec, tokens: np.ndarray) -> None:
        """
        Persist one clip. Tokens land before the spec, so a torn write leaves an
        orphan array rather than a spec pointing at nothing.

        Args:
          spec (ClipSpec): the clip's recipe.
          tokens (np.ndarray): (T, R) codes; cast to int16.
        """
        codes = np.ascontiguousarray(tokens, dtype=np.int16)
        if codes.ndim != 2:
            raise ValueError(f"tokens must be (T, R), got {codes.shape}")
        np.save(self.items_dir / f"{spec.item_id}.npy", codes)
        (self.items_dir / f"{spec.item_id}.json").write_text(
            json.dumps(spec.to_json(), indent=2)
        )
        if spec.item_id not in self._specs:
            self._order.append(spec.item_id)
        self._specs[spec.item_id] = spec

    # -- measured quality ----------------------------------------------------

    @property
    def fills_path(self) -> Path:
        """
        Returns:
          Path: the sidecar holding each clip's measured fill fraction.
        """
        return self.root / "fills.json"

    def _load_fills(self) -> dict[str, float]:
        """
        Returns:
          dict[str, float]: item id to fill fraction, empty when never written.
        """
        if self._fills is None:
            if self.fills_path.exists():
                self._fills = {
                    str(k): float(v)
                    for k, v in json.loads(self.fills_path.read_text()).items()
                }
            else:
                self._fills = {}
        return self._fills

    def set_fill(self, item_id: str, fill: float) -> None:
        """
        Record how full of audio a clip turned out to be.

        Kept beside the specs rather than inside them: a spec is the recipe, and
        fill is an outcome of running it. Persisting it means a clip that came
        out as dead air can be skipped in a later session without decoding it
        again to find out.

        Args:
          item_id (str): the clip id.
          fill (float): fraction above the near-silence floor, in [0, 1].
        """
        fills = self._load_fills()
        fills[item_id] = float(fill)
        self.fills_path.write_text(json.dumps(fills, indent=0, sort_keys=True))

    def fill(self, item_id: str) -> float | None:
        """
        Args:
          item_id (str): the clip id.

        Returns:
          float | None: the recorded fill fraction, or None if never measured.
        """
        return self._load_fills().get(item_id)

    def has(self, item_id: str) -> bool:
        """
        Args:
          item_id (str): the clip id.

        Returns:
          bool: True if both the spec and its tokens are stored.
        """
        return item_id in self._specs and (self.items_dir / f"{item_id}.npy").exists()

    def tokens(self, item_id: str) -> np.ndarray:
        """
        Args:
          item_id (str): the clip id.

        Returns:
          np.ndarray: (T, R) int16 codes.
        """
        path = self.items_dir / f"{item_id}.npy"
        if not path.exists():
            raise KeyError(f"no tokens for item {item_id!r} in {self.root}")
        return np.load(path)

    def spec(self, item_id: str) -> ClipSpec:
        """
        Args:
          item_id (str): the clip id.

        Returns:
          ClipSpec: the stored spec.
        """
        if item_id not in self._specs:
            raise KeyError(f"no spec for item {item_id!r} in {self.root}")
        return self._specs[item_id]

    def specs(self, tier: Tier | None = None) -> list[ClipSpec]:
        """
        Args:
          tier (Tier | None): restrict to one tier, or None for all.

        Returns:
          list[ClipSpec]: stored specs in insertion order.
        """
        specs = [self._specs[i] for i in self._order]
        return [s for s in specs if tier is None or s.tier == tier]

    def groups(self, tier: Tier | None = None) -> dict[str, list[ClipSpec]]:
        """
        Args:
          tier (Tier | None): restrict to one tier, or None for all.

        Returns:
          dict[str, list[ClipSpec]]: specs bucketed by group_id.
        """
        buckets: dict[str, list[ClipSpec]] = {}
        for spec in self.specs(tier):
            buckets.setdefault(spec.group_id, []).append(spec)
        return buckets
