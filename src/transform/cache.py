"""Generation cache (§5.2). Mandatory, not an optimization.

Two independent reasons, both correctness rather than cost. Generation is sampled at
temperature 0.7, so a rerun without a cache would produce *different queries* and silently
change what every downstream number refers to. And §9 requires a run be re-executable; that is
only true if the query set is stable.

The key carries every field that can change the output. `prompt_version` is a human-readable
label, but the template's content hash goes in too -- editing a prompt file without bumping its
version is exactly the mistake that would otherwise reuse stale generations under a new
intent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "generations"


def flatten_id(model_id: str) -> str:
    """Make a provider model id safe as one path segment.

    Open-weight ids carry an org prefix (`meta-llama/llama-3.3-70b-instruct`). Replaced rather
    than stripped so two models differing only in org cannot collide onto one file.
    """
    return model_id.replace("/", "__")


@dataclass(frozen=True)
class GenerationKey:
    dataset: str
    qid: str
    condition: str
    generator: str
    prompt_version: str
    prompt_sha: str
    temperature: float
    seed: int
    # Part of the key: it changes the completion, so a bump must invalidate rather than reuse.
    max_output_tokens: int = 0

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class Generation:
    """One transformed query, with enough provenance to audit it later (§5.6)."""

    text: str
    raw: str
    model_version: str
    key: dict


class GenerationCache:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: GenerationKey) -> Path:
        # Sharded by condition so a directory listing stays navigable by hand during audit.
        shard = self.cache_dir / key.dataset / key.condition
        shard.mkdir(parents=True, exist_ok=True)
        return shard / f"{key.digest()}.json"

    def get(self, key: GenerationKey) -> Generation | None:
        path = self._path(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        return Generation(**payload)

    def put(self, key: GenerationKey, generation: Generation) -> None:
        self._path(key).write_text(json.dumps(asdict(generation), indent=2))

    def stats(self, dataset: str, condition: str) -> int:
        shard = self.cache_dir / dataset / condition
        return len(list(shard.glob("*.json"))) if shard.exists() else 0
