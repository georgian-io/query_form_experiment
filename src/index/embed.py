"""Embedder registry: one interface, three provider routes (§5.3).

The asymmetry this module exists to get right is that queries and documents are *not* encoded
the same way. Every model on the slate is instruction-tuned and expects to be told which side of
the retrieval it is embedding -- as a text prefix for open-weight models, as a task-type enum for
hosted APIs. Getting this wrong costs several NDCG points and produces no error, which is why
`EmbeddingModelConfig.card_verified` must be set before a model can encode at all (§10).

Phase 0 wires the Vertex route end to end. Novita and the other hosted APIs raise until their
model cards have been transcribed -- deliberately, so a stub cannot be mistaken for a working
path and silently poison a leaderboard.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import EmbeddingModelConfig, Provider

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "embeddings"

TextKind = Literal["query", "doc"]


class Embedder(ABC):
    """Encodes text for one configured model.

    `encode` takes the side of the retrieval explicitly rather than defaulting, because a default
    here would be a silent wrong answer for every instruction-tuned model on the slate.
    """

    def __init__(self, config: EmbeddingModelConfig) -> None:
        if not config.card_verified:
            raise ValueError(
                f"{config.id}: refusing to encode -- prompting/pooling have not been transcribed "
                "from the model card and checked. Set card_verified once they have (§5.3)."
            )
        self.config = config

    @abstractmethod
    def _encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray: ...

    def encode(
        self, texts: Sequence[str], kind: TextKind, batch_size: int | None = None
    ) -> np.ndarray:
        """Instruct, batch, encode, and check the shape.

        The instruction is applied here rather than inside each `_encode` so that no provider
        route can forget it -- a dropped query prefix produces no error and costs several NDCG
        points, which is the §10 failure this whole module is arranged to prevent.
        """
        if not texts:
            return np.zeros((0, self.config.dim), dtype=np.float32)

        batch_size = batch_size or self.config.batch_size
        instructed = [self.instruct(text, kind) for text in texts]
        # Hosted APIs 400 on empty input (Voyage/OpenAI/Cohere/Vertex all reject ""); a handful of
        # CORD-19 docs are title-and-abstract-empty. Substitute a single space for those -- hosted
        # only, because sentence-transformers embeds "" fine and changing local input would perturb
        # the published anchors we already reproduced. Such docs are non-relevant, so the
        # placeholder vector is inert to NDCG. The cache key is computed upstream from the original
        # text, so this does not fork the cache.
        if self.config.provider is not Provider.LOCAL:
            instructed = [text if text.strip() else " " for text in instructed]
        batches = [self._encode(list(b), kind) for b in _batched(instructed, batch_size)]
        vectors = np.vstack(batches).astype(np.float32)

        if vectors.shape != (len(texts), self.config.dim):
            raise ValueError(
                f"{self.config.id}: got {vectors.shape}, expected "
                f"({len(texts)}, {self.config.dim}) -- config dim disagrees with the provider"
            )
        return _l2_normalize(vectors) if self.config.normalize else vectors

    def instruct(self, text: str, kind: TextKind) -> str:
        """Apply the model card's text instruction. No-op for providers that use task types."""
        prefix = self.config.query_instruction if kind == "query" else self.config.doc_instruction
        return f"{prefix}{text}" if prefix else text

    def cache_key(self, text: str, kind: TextKind) -> str:
        """Keyed on every version field, so a revision or instruction bump invalidates precisely."""
        payload = json.dumps(
            {
                "model": self.config.id,
                "api_model": self.config.model_name,
                "revision": self.config.revision,
                "kind": kind,
                "instruction": self.config.query_instruction
                if kind == "query"
                else self.config.doc_instruction,
                "task_type": self.config.query_task_type
                if kind == "query"
                else self.config.doc_task_type,
                "dim": self.config.dim,
                "dtype": self.config.embd_dtype,
                "batch_size": self.config.batch_size,
                "text": text,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class VertexEmbedder(Embedder):
    """`gemini-embedding-001` via Vertex AI.

    Vertex encodes the query/document distinction as `task_type` (RETRIEVAL_QUERY vs
    RETRIEVAL_DOCUMENT) rather than a text prefix, and pools server-side -- so `pooling` and the
    instruction fields are meaningless here and the config validator forbids them.
    """

    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__(config)
        from google import genai

        self._genai = genai
        self._client = genai.Client(
            vertexai=True,
            project=_require_env("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray:
        task_type = (
            self.config.query_task_type if kind == "query" else self.config.doc_task_type
        )
        config = self._genai.types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.config.dim,
        )
        # gemini-embedding-001 truncates over-long inputs server-side and has no per-request token
        # cap, so it stays a single call (preserving its exact reproduction). The gecko
        # text-embedding-* models instead 400 when an input exceeds max_seq_len OR a request sums
        # past ~20k tokens, so cap each input and pack requests by token budget.
        if self.config.model_name.startswith("text-embedding-0"):
            # Budget in cl100k tokens, but Vertex counts with its own tokenizer -- observed ~16%
            # higher (an 18k-cl100k request measured 20883 on Vertex, over its 20k cap). 13k leaves
            # margin for ratios up to ~1.5x.
            groups = _token_batches(_truncate_to_tokens(texts, self.config.max_seq_len), 13000)
        else:
            groups = [list(texts)]
        vectors: list = []
        for group in groups:
            response = self._client.models.embed_content(
                model=self.config.model_name, contents=group, config=config
            )
            vectors.extend(e.values for e in response.embeddings)
        return np.array(vectors, dtype=np.float32)


class _StubEmbedder(Embedder):
    """Placeholder for a route whose model cards have not been transcribed yet."""

    _hint = ""

    def _encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray:
        raise NotImplementedError(
            f"{self.config.provider} route is not wired yet ({self.config.id}). {self._hint}"
        )


class VoyageEmbedder(Embedder):
    """Voyage AI, e.g. `voyage-4-large`.

    Voyage offers two equivalent ways to state the query/document distinction, and using both
    double-prefixes. Passing `input_type="query"` makes Voyage prepend "Represent the query for
    retrieving supporting documents: " itself; RTEB instead passes `input_type=None` and
    concatenates that exact string by hand. We take the `input_type` route, which is why the
    config carries task types and leaves `query_instruction` empty. Do not "fix" this to match
    RTEB's source literally without also clearing the task types -- the doubled form embeds
    cleanly and merely scores worse.

    `output_dimension` must be passed explicitly: Voyage defaults voyage-4-large to 1024, while
    the leaderboard row is 2048.

    Batch size affects reproducibility, though not meaning. Inputs are embedded independently,
    but GPU reduction order varies with batch shape, and at bf16-class precision that lands
    around 1e-3 per element. Measured here: batch >= 2 is bit-identical across repeated calls,
    batch == 1 is not even self-reproducible, and the two shapes differ enough to reorder a
    sub-0.01 margin. Config pins the value and forbids 1.
    """

    # Voyage spells float32 as plain "float"; the other dtypes match our config vocabulary.
    _DTYPES = {"float32": "float"}

    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__(config)
        import voyageai

        self._client = voyageai.Client(api_key=_require_env("VOYAGE_API_KEY"))

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray:
        input_type = self.config.query_task_type if kind == "query" else self.config.doc_task_type
        dtype = self.config.embd_dtype
        response = self._client.embed(
            list(texts),
            model=self.config.model_name,
            input_type=input_type,
            truncation=True,
            output_dimension=self.config.dim,
            output_dtype=self._DTYPES.get(dtype, dtype),
        )
        return np.array(response.embeddings, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    """OpenAI, e.g. `text-embedding-3-large`.

    The odd one out on the slate: OpenAI has no query/document asymmetry at all -- no task type,
    no input type, no prefix. RTEB passes it nothing either. `dimensions` is sent explicitly
    because the leaderboard lists 3072d and 512d as separate rows.
    """

    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__(config)
        from openai import OpenAI

        self._client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray:
        # OpenAI 400s on inputs over 8192 tokens and does not truncate server-side (unlike
        # Voyage/Cohere/Vertex). cl100k is text-embedding-3's own tokenizer, so this truncates
        # exactly the docs that would be rejected -- long CORD-19 abstracts -- and leaves shorter
        # corpora untouched and reproducible.
        response = self._client.embeddings.create(
            model=self.config.model_name,
            input=_truncate_to_tokens(texts, self.config.max_seq_len),
            dimensions=self.config.dim,
        )
        # `index` is returned explicitly by the API; sorting on it rather than trusting order.
        ordered = sorted(response.data, key=lambda d: d.index)
        return np.array([d.embedding for d in ordered], dtype=np.float32)


class CohereEmbedder(Embedder):
    """Cohere, e.g. `embed-v4.0`.

    Cohere's `input_type` is mandatory from v3 onward and is the query/document selector, so it
    maps onto our task-type fields. Requesting only the `float` embedding type keeps the response
    aligned with `embd_dtype`; the quantized types are separate leaderboard rows.
    """

    _DTYPES = {"float32": "float"}

    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__(config)
        import cohere

        self._client = cohere.ClientV2(api_key=_require_env("COHERE_API_KEY"))

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray:
        input_type = self.config.query_task_type if kind == "query" else self.config.doc_task_type
        dtype = self._DTYPES.get(self.config.embd_dtype, self.config.embd_dtype)
        response = self._client.embed(
            model=self.config.model_name,
            texts=list(texts),
            input_type=input_type,
            embedding_types=[dtype],
            output_dimension=self.config.dim,
        )
        return np.array(getattr(response.embeddings, dtype), dtype=np.float32)


class LocalEmbedder(Embedder):
    """sentence-transformers, run in-process. Used for the §7 contrast anchors.

    This is the only route where we control the forward pass, which buys two things the hosted
    endpoints cannot give us:

    * **A pinnable revision.** `revision` is passed to the loader, so the weights are fixed by
      commit hash. Every hosted model on the slate is a moving target -- that is precisely what
      made the voyage-4-large residual irreducible.
    * **Explicit pooling.** The config's `pooling` is asserted against what the loaded module
      stack actually does, rather than trusted. A model card saying "CLS" and a checkpoint
      configured for mean pooling is a silent several-point error, and §5.3 puts pooling second
      only to instructions in the diagnosis order.

    Normalization is left to `Embedder.encode` via `config.normalize`, so the same rule applies
    across every provider.
    """

    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__(config)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            config.model_name,
            revision=config.revision,
            device=_local_device(),
        )
        self._assert_pooling_matches_card()

    def _assert_pooling_matches_card(self) -> None:
        pooling_modules = [
            m for m in self._model._modules.values() if type(m).__name__ == "Pooling"
        ]
        if not pooling_modules:
            raise ValueError(f"{self.config.id}: no pooling module found in the loaded stack")

        actual = str(pooling_modules[0].pooling_mode)
        expected = {"mean": "mean", "cls": "cls", "last-token": "lasttoken"}[self.config.pooling]
        if expected not in actual.replace("_", "").lower():
            raise ValueError(
                f"{self.config.id}: config says pooling={self.config.pooling!r} but the "
                f"checkpoint pools {actual!r} -- one of the two is wrong, and this fails silently"
            )

    def _encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray:
        # normalize_embeddings stays False: Embedder.encode owns that, per config.normalize.
        return self._model.encode(
            list(texts),
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)


def _local_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class NovitaEmbedder(Embedder):
    """Open-weight models served behind Novita's OpenAI-compatible endpoint.

    Novita runs the forward pass, so pooling and normalization are its business, not ours -- the
    config validator forbids `pooling` here for that reason. What we *do* control is the
    instruction text, which for these models is part of the input string rather than a separate
    parameter: Qwen3-Embedding-8B expects `Instruct: {task}\\nQuery:{query}` on the query side and
    bare text on the document side.

    No revision can be pinned; Novita exposes none. Record the run date (§9).
    """

    _BASE_URL = "https://api.novita.ai/v3/openai"

    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__(config)
        from openai import OpenAI

        self._client = OpenAI(
            api_key=_require_env("NOVITA_API_KEY"), base_url=self._BASE_URL
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray:
        # Novita's OpenAI-compatible endpoint also rejects over-limit inputs. Qwen tokenizes
        # differently from cl100k, so truncate with headroom below its 8192-token limit.
        max_tokens = int(self.config.max_seq_len * 0.85) if self.config.max_seq_len else None
        response = self._client.embeddings.create(
            model=self.config.model_name,
            input=_truncate_to_tokens(texts, max_tokens),
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda d: d.index)
        return np.array([d.embedding for d in ordered], dtype=np.float32)


class APIEmbedder(_StubEmbedder):
    _hint = "Add the provider client (OpenAI / Cohere) and its retrieval input type."


# Checked before the provider default, because `api` covers several vendors with incompatible
# clients. Prefix-keyed so model families route without a per-model entry.
_MODEL_ROUTES: dict[str, type[Embedder]] = {
    "voyage-": VoyageEmbedder,
    "text-embedding-3": OpenAIEmbedder,
    # Cohere names every embedding model `embed-*` (embed-v4.0, embed-multilingual-v3.0).
    "embed-": CohereEmbedder,
}

_PROVIDER_ROUTES: dict[Provider, type[Embedder]] = {
    Provider.VERTEX: VertexEmbedder,
    Provider.NOVITA: NovitaEmbedder,
    Provider.LOCAL: LocalEmbedder,
    Provider.API: APIEmbedder,
}


def get_embedder(config: EmbeddingModelConfig) -> Embedder:
    """Pick a client. Routes on `model_name`, not `id`.

    It is the provider's own model string that determines which client can serve it -- `id` is
    ours to choose, and quantized rows deliberately give it a suffix (`embed-v4.0-int8-512`)
    that no vendor would recognise.
    """
    for prefix, route in _MODEL_ROUTES.items():
        if config.model_name.startswith(prefix):
            return route(config)
    return _PROVIDER_ROUTES[config.provider](config)


# How many docs to encode before flushing them to the on-disk cache. Bounds how much a killed run
# re-does: a large corpus resumes within one chunk of where it stopped rather than from zero.
_CACHE_CHUNK = 512


class CachedEmbedder:
    """Wraps an `Embedder` with an on-disk vector cache (§5.3, §9).

    Caching is a correctness requirement, not an optimization: reruns of the reproduction gate
    must not re-bill or re-sample the provider. Only cache misses reach the network.
    """

    def __init__(self, embedder: Embedder, cache_dir: Path | None = None) -> None:
        self.embedder = embedder
        self.cache_dir = (cache_dir or CACHE_DIR) / embedder.config.id
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def encode(
        self, texts: Sequence[str], kind: TextKind, batch_size: int | None = None
    ) -> np.ndarray:
        keys = [self.embedder.cache_key(t, kind) for t in texts]
        cached = {k: v for k in set(keys) if (v := self._read(k)) is not None}

        missing = [t for t, k in zip(texts, keys, strict=True) if k not in cached]
        # Encode and cache in chunks rather than all-then-cache, so an interrupted run resumes from
        # the last cached chunk instead of re-billing a large corpus from zero. Caching is mandatory
        # (non-negotiable #3); making it incremental is what keeps slow/large builds convergent when
        # the environment reaps long jobs. Same cache files and keys -- only the write cadence
        # changes -- so it cannot alter a vector, only how much progress a kill costs (<= one chunk).
        for start in range(0, len(missing), _CACHE_CHUNK):
            chunk = missing[start : start + _CACHE_CHUNK]
            fresh = self.embedder.encode(chunk, kind, batch_size=batch_size)
            for text, vector in zip(chunk, fresh, strict=True):
                key = self.embedder.cache_key(text, kind)
                self._write(key, vector)
                cached[key] = vector

        return np.vstack([cached[k] for k in keys]).astype(np.float32)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def _read(self, key: str) -> np.ndarray | None:
        path = self._path(key)
        return np.load(path) if path.exists() else None

    def _write(self, key: str, vector: np.ndarray) -> None:
        np.save(self._path(key), vector)


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


_TIKTOKEN_ENC = None


def _encoder():
    """cl100k_base tokenizer, lazily loaded once. A good proxy for the hosted models' own
    tokenizers -- exact for OpenAI, close enough elsewhere to keep requests under provider caps."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        import tiktoken

        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENC


def _truncate_to_tokens(texts: Sequence[str], max_tokens: int | None) -> list[str]:
    """Truncate each text to at most `max_tokens` cl100k tokens.

    For providers that reject over-limit inputs instead of truncating server-side (OpenAI,
    Novita, and the Vertex gecko models). Only texts that actually exceed the limit are re-encoded,
    so this is a no-op for short corpora and does not disturb their cached vectors. `max_tokens=None`
    disables it.
    """
    if max_tokens is None:
        return list(texts)
    enc = _encoder()
    out: list[str] = []
    for text in texts:
        tokens = enc.encode(text)
        out.append(enc.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text)
    return out


def _token_batches(texts: Sequence[str], max_tokens: int) -> list[list[str]]:
    """Pack texts into groups whose total cl100k token count stays under `max_tokens`.

    Vertex's gecko text-embedding models reject a *request* summing past ~20k tokens, so batching
    by count overflows on long-doc corpora. Callers truncate each input below max_tokens first, so
    every text fits in a group by itself.
    """
    enc = _encoder()
    groups: list[list[str]] = []
    current: list[str] = []
    running = 0
    for text in texts:
        n = len(enc.encode(text))
        if current and running + n > max_tokens:
            groups.append(current)
            current, running = [], 0
        current.append(text)
        running += n
    if current:
        groups.append(current)
    return groups


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; see .env.example")
    return value
