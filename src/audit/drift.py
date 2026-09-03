"""§5.6 intent-drift audit: did a query transform change WHAT is being asked?

A transformed query has DRIFTED if the set of documents that correctly answer it differs from the
original's — the transform changed the information need, not just the phrasing. Drift moves ΔNDCG
for reasons unrelated to embedding quality (a "worse" score may be a genuinely different question),
so §1.3 only lets us headline a condition whose drift rate is below a pre-registered threshold.

Two design choices mirror the rest of the study:
- **Cross-lineage judging.** A rewrite is never judged by a model of its own generator's lineage,
  so we do not measure a model's preference for its own outputs (the §1.2 concern, applied to the
  audit itself).
- **Versioned rubric.** The prompt is content-hashed into the cache key, so editing the rubric
  without bumping its version cannot silently reuse stale verdicts (the §5.2 cache discipline).

Drift is a property of the (original, transformed) query pair alone — it does not depend on any
embedding model or ranking — so this audit runs on the query sets directly, with no dependency on
retrieval results.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import GeneratorConfig
from src.transform.generators import get_generator

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "audit"

# §1.3 go/no-go: a condition is headline-able only if its drift rate is below this. Pinned here as
# the pre-registration artifact -- fixed before any verdict is read.
DRIFT_THRESHOLD = 0.15

VERDICTS = ("PRESERVED", "DRIFTED", "UNCERTAIN")

# Cross-lineage judge for each generator lineage: never a model of the generator's own lineage.
# Keyed by the generator's `lineage` field (google/anthropic/openai/meta), value is a judge id.
JUDGE_FOR_LINEAGE: dict[str, str] = {
    "google": "claude-opus-4-8",       # anthropic judges google
    "anthropic": "gemini-2.5-pro",     # google judges anthropic
    "openai": "gemini-2.5-pro",        # google judges openai
    "meta": "gemini-2.5-pro",          # google judges meta
}


def _flat(s: str) -> str:
    return s.replace("/", "__")


@dataclass(frozen=True)
class AuditKey:
    dataset: str
    qid: str
    condition: str
    generator: str
    judge: str
    rubric_version: str
    rubric_sha: str

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Verdict:
    verdict: str  # PRESERVED | DRIFTED | UNCERTAIN
    reason: str
    judge: str
    model_version: str
    key: dict

    @property
    def drifted(self) -> bool:
        return self.verdict == "DRIFTED"


class AuditCache:
    """Verdict cache, sharded by dataset/condition like the generation cache (§5.2)."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: AuditKey) -> Path:
        shard = self.cache_dir / key.dataset / key.condition
        shard.mkdir(parents=True, exist_ok=True)
        return shard / f"{key.digest()}.json"

    def get(self, key: AuditKey) -> Verdict | None:
        path = self._path(key)
        if not path.exists():
            return None
        return Verdict(**json.loads(path.read_text()))

    def put(self, key: AuditKey, verdict: Verdict) -> None:
        self._path(key).write_text(json.dumps(asdict(verdict), indent=2))


def _parse_verdict(text: str) -> tuple[str, str]:
    """Pull {verdict, reason} out of a judge completion; tolerate prose around the JSON."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            v = str(obj.get("verdict", "")).strip().upper()
            if v in VERDICTS:
                return v, str(obj.get("reason", ""))[:300]
        except json.JSONDecodeError:
            pass
    # Fallback: a bare keyword anywhere in the reply.
    upper = text.upper()
    for v in ("DRIFTED", "PRESERVED", "UNCERTAIN"):
        if v in upper:
            return v, text.strip()[:300]
    raise ValueError(f"judge returned no parseable verdict: {text[:200]!r}")


def sample_qids(qids: list[str], n: int, seed: int) -> list[str]:
    """Seeded sample so every generator is audited on the SAME queries; all of them if n >= total."""
    ordered = sorted(qids)
    if n >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, n))


class IntentDriftAuditor:
    def __init__(self, judges: dict[str, GeneratorConfig], temperature: float, seed: int,
                 rubric_version: str = "v1", prompts_dir: Path | None = None) -> None:
        self._dir = prompts_dir or PROMPTS_DIR
        self.rubric_version = rubric_version
        self._template = (self._dir / f"intent_drift.{rubric_version}.md").read_text()
        self.rubric_sha = hashlib.sha256(self._template.encode()).hexdigest()[:16]
        # judge id -> constructed Generator, built lazily and reused
        self._judge_configs = judges
        self._judge_temp = temperature
        self._judge_seed = seed
        self._clients: dict[str, object] = {}
        self._cache = AuditCache()

    def _client(self, judge_id: str):
        if judge_id not in self._clients:
            cfg = self._judge_configs[judge_id]
            self._clients[judge_id] = get_generator(cfg, self._judge_temp, self._judge_seed)
        return self._clients[judge_id]

    def judge_id_for(self, generator_lineage: str) -> str:
        return JUDGE_FOR_LINEAGE[generator_lineage]

    def audit_one(self, *, dataset: str, qid: str, condition: str, generator: str,
                  generator_lineage: str, original: str, transformed: str) -> Verdict:
        judge_id = self.judge_id_for(generator_lineage)
        key = AuditKey(dataset=dataset, qid=qid, condition=condition, generator=generator,
                       judge=judge_id, rubric_version=self.rubric_version, rubric_sha=self.rubric_sha)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        prompt = self._template.format(original=original, transformed=transformed)
        # A judge that fails on one query -- an empty completion after retries, or a provider
        # safety filter on sensitive patient text -- must not crash the batch. Record it as
        # UNCERTAIN (which drift_rate excludes from the denominator, so it neither counts as drift
        # nor dilutes the rate) with the cause in `reason`, and move on. The count of judge errors
        # is visible in the results, so a condition drowning in them is not silently "clean".
        try:
            completion = self._client(judge_id).generate(prompt)
            verdict, reason = _parse_verdict(completion.text)
            model_version = completion.model_version
        except Exception as exc:  # noqa: BLE001 -- boundary: isolate one query's judge failure
            verdict, reason = "UNCERTAIN", f"JUDGE_ERROR: {type(exc).__name__}: {exc}"[:300]
            model_version = f"JUDGE_ERROR::{judge_id}"
        result = Verdict(verdict=verdict, reason=reason, judge=judge_id,
                         model_version=model_version, key=key.__dict__)
        self._cache.put(key, result)
        return result


def drift_rate(verdicts: list[Verdict]) -> dict:
    """Flagged rate = DRIFTED / (judged, excluding UNCERTAIN). Reports the raw counts too."""
    n = len(verdicts)
    drifted = sum(1 for v in verdicts if v.verdict == "DRIFTED")
    uncertain = sum(1 for v in verdicts if v.verdict == "UNCERTAIN")
    judge_errors = sum(1 for v in verdicts if v.reason.startswith("JUDGE_ERROR:"))
    denom = n - uncertain
    rate = drifted / denom if denom else 0.0
    return {
        "n": n,
        "drifted": drifted,
        "preserved": n - drifted - uncertain,
        "uncertain": uncertain,
        "judge_errors": judge_errors,  # subset of uncertain: judge couldn't answer (filter/error)
        "drift_rate": rate,
        "below_threshold": rate < DRIFT_THRESHOLD,
        "threshold": DRIFT_THRESHOLD,
    }
