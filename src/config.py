"""Pydantic-validated loaders for `config/experiment.yaml` and `config/datasets.yaml`.

Malformed or under-specified config must fail here, at startup, rather than halfway through a
paid embedding run. The validators below encode the §5.3/§9 requirements that are easy to get
silently wrong -- an unverified query instruction, an implicit tokenizer, an unpinned revision --
because those are precisely the mistakes the reproduction gate would otherwise surface as a
mysterious NDCG delta.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"

# Load `.env` on import so scripts and tests see the same environment without a wrapper. Real
# environment variables win (python-dotenv does not override by default), which keeps CI and
# one-off `VAR=... uv run ...` invocations authoritative over the file.
load_dotenv(REPO_ROOT / ".env")


class Provider(StrEnum):
    NOVITA = "novita"
    VERTEX = "vertex"
    API = "api"
    # Generator-only providers: these serve chat models for query transformation (§5.2), not
    # embeddings, so they never appear on the embedding slate.
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    # Open models (Llama) served through Vertex AI Model-as-a-Service on the OpenAI-compatible
    # `openapi/chat/completions` path. Distinct from VERTEX (the Gemini SDK route): different
    # transport, different auth surface, same ADC. Added when Novita's Llama endpoint went into
    # sustained 429 `server_overload` mid-study -- see the 2026-07-23 NOTES entry.
    VERTEX_MAAS = "vertex_maas"
    # sentence-transformers in-process. The only route where we control the forward pass, and
    # therefore the only one where `pooling` is actionable rather than documentation. Also the
    # only route with a genuinely pinnable revision (§9): every hosted endpoint on this slate is
    # a moving target, which is what made the voyage-4-large residual irreducible.
    LOCAL = "local"


class Pooling(StrEnum):
    MEAN = "mean"
    CLS = "cls"
    LAST_TOKEN = "last-token"


class Similarity(StrEnum):
    COSINE = "cosine"
    DOT = "dot"


class RetrievalMode(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class Condition(StrEnum):
    PARAPHRASE = "paraphrase"
    VERBOSE = "verbose"
    TERSE = "terse"
    HYDE = "hyde"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EmbeddingModelConfig(_Strict):
    """One row of the §7 model slate.

    Carries both prompting shapes because the slate mixes them: open-weight models take literal
    text instructions plus an explicit pooling strategy, while hosted APIs express the same
    query/document asymmetry as a task-type enum and pool internally.
    """

    id: str
    # The string sent to the provider, when it differs from `id`. RTEB lists quantized variants
    # as separate leaderboard rows served by the *same* endpoint -- `embed-v4.0` appears at both
    # 1536/float32 and 512/int8 -- so `id` stays unique for namespaces and results while
    # `model_name` is what the API sees.
    api_model_name: str | None = None
    provider: Provider
    family: str
    dim: int
    revision: str | None = None
    max_seq_len: int | None = None
    similarity: Similarity = Similarity.COSINE
    normalize: bool = True
    # Output precision. Pinned because it is part of a model's identity on this leaderboard, not
    # a storage detail -- RTEB lists "voyage-3.5 (int8, 512d)" as a separate row from voyage-3.5,
    # and quantized output changes the vectors and therefore the ranking.
    embd_dtype: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32"
    # Pinned for reproducibility, not because batching changes meaning. Texts are embedded
    # independently -- there is no cross-attention between inputs -- but GPU reduction order
    # varies with batch shape, and at bf16-class serving precision that shows up as ~1e-3 per
    # element rather than the ~1e-7 fp32 intuition suggests. Measured on voyage-4-large:
    # batch >= 2 is bit-identical across repeated calls, batch == 1 is NOT even self-reproducible
    # (maxabs 5e-3 between repeats), and the two shapes differ reproducibly enough to reorder a
    # contest whose margin is under ~0.01. Semantically irrelevant; fatal to exact reproduction.
    # Hence: folded into the embedding cache key, and 1 is forbidden.
    batch_size: int = Field(default=8, ge=2)

    # Text instructions prepended to the input (§5.3), verbatim from the model card. NOT limited
    # to open-weight models: RTEB prefixes voyage-4-large with "Represent the query for
    # retrieving supporting documents: " while passing gemini-embedding-001 no prefix at all.
    query_instruction: str | None = None
    doc_instruction: str | None = None

    # How token vectors collapse to one embedding. Only meaningful where we control the forward
    # pass -- hosted APIs pool internally and expose no such knob.
    pooling: Pooling | None = None

    # Hosted-API task selector -- e.g. gemini's RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT, or
    # Voyage's input_type. Independent of the instruction: Voyage takes both.
    query_task_type: str | None = None
    doc_task_type: str | None = None

    # Whether prompting/pooling were transcribed from the model card and checked. Encoding is
    # refused while this is false: a wrong or missing query prefix is the single most common
    # cause of failing the reproduction gate (§5.3, §10), and it fails *quietly*.
    card_verified: bool = False

    @property
    def model_name(self) -> str:
        """What to send the provider. Defaults to `id`."""
        return self.api_model_name or self.id

    @model_validator(mode="after")
    def _prompting_shape_matches_provider(self) -> EmbeddingModelConfig:
        """Reject config a provider would silently ignore.

        Three distinctions, deliberately kept separate because conflating them is what produced
        the earlier round of bugs:

        * `pooling` is actionable only where we run the forward pass -- i.e. `local`. Novita
          serves open-weight models behind an OpenAI-compatible endpoint and pools server-side,
          so setting it there would be inert config that reads as if it were doing something.
        * task types exist only on hosted APIs that model the query/document split as an enum
          (Vertex, Voyage, Cohere). Neither local nor Novita has them.
        * instructions are unconstrained -- any provider may take a text prefix.
        """
        if self.pooling is not None and self.provider is not Provider.LOCAL:
            raise ValueError(
                f"{self.id}: pooling needs control of the forward pass; {self.provider} pools "
                "server-side, so setting it here would be inert"
            )
        if self.card_verified and self.provider is Provider.LOCAL and self.pooling is None:
            raise ValueError(f"{self.id}: card_verified requires an explicit pooling strategy")

        has_task_type = self.query_task_type is not None or self.doc_task_type is not None
        if has_task_type and self.provider in (Provider.NOVITA, Provider.LOCAL):
            raise ValueError(
                f"{self.id}: task types are a hosted-API concept; {self.provider} models express "
                "the query/document distinction through query_instruction/doc_instruction"
            )
        return self


class GeneratorConfig(_Strict):
    id: str
    provider: Provider
    lineage: str
    revision: str | None = None
    # The model string the API expects, when it differs from our lineage `id`. Vertex MaaS names
    # Llama `meta/llama-3.3-70b-instruct-maas`, but the lineage id stays the Novita-style
    # `meta-llama/llama-3.3-70b-instruct` so result filenames and the cross-generator board are
    # unchanged by the route switch. None = send `id` verbatim.
    api_model_name: str | None = None
    # §7 pins temperature at 0.7, but not every model still accepts it. Anthropic removed
    # `temperature` on Opus 4.7+ (it returns a 400), and OpenAI's reasoning models reject
    # non-default values. Those models are sampled at their own default instead, which is a real
    # asymmetry across the generator slate rather than something we can normalize away -- record
    # it here so a cross-generator comparison is read with that in mind.
    supports_temperature: bool = True
    # Pinned per §9 and folded into the generation cache key, because it changes the OUTPUT, not
    # just the ceiling. On Vertex it also counts thinking tokens for the Gemini 2.5 family, which
    # is how 2048 silently truncated 44 of 50 AILAStatutes rewrites into what looked like
    # deliberate summaries. Sized for the longest AILAStatutes narrative (~5.9k chars).
    max_output_tokens: int = 8192
    # Gemini 2.5 Pro requires a non-zero thinking budget (0 is rejected), so thinking is pinned
    # to the minimum rather than disabled -- a paraphrase needs no reasoning trace, and on Vertex
    # these tokens are charged against max_output_tokens. None = leave the provider's default.
    thinking_budget: int | None = None


class ExperimentConfig(_Strict):
    embedding_models: list[EmbeddingModelConfig]
    retrieval_modes: list[RetrievalMode]
    generators: list[GeneratorConfig] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    temperature: float = 0.7
    k: int = 10
    retrieve_depth: int = 100
    # turbopuffer's server-side RRF constant. We do not pass it -- `rerank_by=("RRF",)` takes the
    # server default -- but 60 was confirmed by offline parity (50/50 queries identical), and
    # pinning it here means the offline cross-check and any future drift in the server default
    # are both detectable rather than silent (§9, §10).
    rrf_k: int = 60
    seed: int = 20260720

    @model_validator(mode="after")
    def _ids_unique(self) -> ExperimentConfig:
        for label, ids in (
            ("embedding_models", [m.id for m in self.embedding_models]),
            ("generators", [g.id for g in self.generators]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate ids in {label}")
        return self

    def model(self, model_id: str) -> EmbeddingModelConfig:
        for m in self.embedding_models:
            if m.id == model_id:
                return m
        known = ", ".join(m.id for m in self.embedding_models)
        raise KeyError(f"unknown embedding model {model_id!r}; configured: {known}")


class TokenizerConfig(_Strict):
    """turbopuffer full-text-search settings.

    `tokenizer` has no default on purpose -- §9 requires it be written out explicitly so an
    upstream default change cannot silently alter BM25 scores between runs.

    The same argument applies to the BM25 scoring constants. turbopuffer fills in `k1`, `b` and
    `max_token_length` server-side when they are omitted, which means omitting them is exactly
    the "rely on the default" that §9 forbids: they scale every BM25 and hybrid score, and a
    change upstream would move our numbers with nothing in the diff to show for it. The defaults
    below were read back off a live namespace rather than guessed.
    """

    tokenizer: Literal["word_v0", "word_v1", "word_v2", "word_v3", "word_v4", "pre_tokenized_array"]
    # turbopuffer defaults this to english even when omitted -- including on code corpora. It is
    # inert while `stemming` and `remove_stopwords` are both false, but pin it rather than
    # inherit it.
    language: Literal["english"] | None = None
    stemming: bool
    remove_stopwords: bool
    case_sensitive: bool = False
    ascii_folding: bool = False
    # BM25 term-frequency saturation and length normalization.
    #
    # `k3` (query-term-frequency saturation, server value 8.0) is deliberately absent: the
    # turbopuffer client does not expose it, so it cannot be pinned from here. It only bites
    # when a term repeats within a single query, which is rare but not impossible for the long
    # AILAStatutes narratives. Recorded as a known unpinned parameter rather than left silent.
    k1: float = 1.2
    b: float = 0.75
    max_token_length: int = 39


class DatasetConfig(_Strict):
    adapter: str
    hf_repo: str
    hf_revision: str | None = None
    # Some RTEB HuggingFace exports ship truncated relevance files, so qrels may have to come
    # from a different repo than corpus/queries. When set, this overrides `hf_repo` for qrels
    # only; the adapter is responsible for asserting the two repos agree on corpus and queries.
    qrels_hf_repo: str | None = None
    qrels_hf_revision: str | None = None
    qrels_path: str = "qrels/test.jsonl"
    split: str = "test"
    # From the dataset card; the §5.1 adapter test asserts the loaded data matches these.
    n_queries: int
    n_docs: int
    n_qrels: int | None = None
    tokenizer: TokenizerConfig
    priority_transforms: list[Condition] = Field(default_factory=list)
    qrel_risk: Literal["low", "medium", "high"] = "medium"
    # Published RTEB NDCG@10 per model id -- the target the §9 gate compares against.
    published_ndcg_at_10: dict[str, float] = Field(default_factory=dict)
    # One line of framing handed to the transform prompt, so a rewrite respects what the queries
    # actually are. Without it a paraphrase of a legal case narrative drifts toward generic prose
    # and the measured effect would be the generator's ignorance of the domain, not phrasing.
    domain_hint: str = ""
    notes: str | None = None


class DatasetsConfig(_Strict):
    datasets: dict[str, DatasetConfig]

    def dataset(self, name: str) -> DatasetConfig:
        if name not in self.datasets:
            known = ", ".join(self.datasets)
            raise KeyError(f"unknown dataset {name!r}; configured: {known}")
        return self.datasets[name]


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


@cache
def load_experiment(path: Path | None = None) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_load_yaml(path or CONFIG_DIR / "experiment.yaml"))


@cache
def load_datasets(path: Path | None = None) -> DatasetsConfig:
    return DatasetsConfig.model_validate(_load_yaml(path or CONFIG_DIR / "datasets.yaml"))


def config_hash(*models: BaseModel) -> str:
    """Stable short hash over config objects, for the provenance field on every result row (§9)."""
    payload = json.dumps([m.model_dump(mode="json") for m in models], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
