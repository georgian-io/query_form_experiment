"""Query-transformation conditions (§5.2).

A `Condition` turns an original query into a prompt, and turns the model's reply back into a
query string. It knows nothing about which LLM produces the text -- that is the generator's job
-- which is what lets generators be swapped for the §1.2 lineage control without touching the
conditions.

Phase 2 implements `paraphrase` only, deliberately. It is the qrel-safe condition: an
intent-preserving restyle should leave the existing relevance judgements valid, so a measured
ΔNDCG is attributable to phrasing rather than to the labels having gone stale. The riskier
conditions (`verbose`, `terse`, `hyde`) come in Phase 3 once the analysis spine is proven.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class Condition(ABC):
    """One transform condition, backed by a versioned prompt template."""

    name: str
    version: str

    def __init__(self, prompts_dir: Path | None = None) -> None:
        self._dir = prompts_dir or PROMPTS_DIR
        self._template = (self._dir / f"{self.name}.{self.version}.md").read_text()

    @property
    def prompt_version(self) -> str:
        return f"{self.name}.{self.version}"

    @property
    def prompt_sha(self) -> str:
        """Content hash of the template.

        Goes into the cache key alongside `prompt_version` so that editing a prompt without
        bumping its version cannot silently reuse generations made under the old wording.
        """
        return hashlib.sha256(self._template.encode()).hexdigest()[:16]

    @abstractmethod
    def build_prompt(self, query: str, domain_hint: str = "") -> str: ...

    def postprocess(self, raw: str) -> str:
        """Clean a reply into a bare query string.

        Models wrap answers in quotes or preambles even when told not to; leaving that in would
        put `"` and `Here is the rewritten query:` into the embedded text and measure the
        wrapper rather than the rewrite.
        """
        text = raw.strip()
        for prefix in ("Rewritten query:", "Query:", "Here is the rewritten query:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
        return text


class Paraphrase(Condition):
    name = "paraphrase"
    version = "v1"

    def build_prompt(self, query: str, domain_hint: str = "") -> str:
        return self._template.format(query=query, domain_hint=domain_hint)


class Terse(Condition):
    """Strip a query to its core information need -- keyword-style, the way a person types.

    Deliberately shortens, so it is exempt from the paraphrase length gate: a low length ratio
    is the intended effect here, not a truncated generation.
    """

    name = "terse"
    version = "v1"

    def build_prompt(self, query: str, domain_hint: str = "") -> str:
        return self._template.format(query=query, domain_hint=domain_hint)


class Verbose(Condition):
    """Expand a query into a full, explicit natural-language information need.

    Models how LLM-authored / conversational-search queries appear relative to terse human
    keyword queries: complete sentences, more descriptive, several times longer (LLM search
    queries run ~23 words vs ~4 for keyword search; see verbose.v1.md's sources). It adds
    explicitness and context, not new constraints, so it is intended to stay close to qrel-safe
    -- the opposite risk profile from terse. Deliberately lengthens, so it is exempt from the
    paraphrase length gate.
    """

    name = "verbose"
    version = "v1"

    def build_prompt(self, query: str, domain_hint: str = "") -> str:
        return self._template.format(query=query, domain_hint=domain_hint)


CONDITIONS: dict[str, type[Condition]] = {
    "paraphrase": Paraphrase,
    "terse": Terse,
    "verbose": Verbose,
}


def get_condition(name: str) -> Condition:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; implemented: {', '.join(CONDITIONS)}")
    return CONDITIONS[name]()
