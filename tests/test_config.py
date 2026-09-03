"""Config-loader tests.

These target the validators that exist to stop a *silent* misconfiguration -- an unpinned
tokenizer, an API model given a text instruction it will ignore -- rather than restating the
schema back at itself.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import (
    DatasetsConfig,
    EmbeddingModelConfig,
    ExperimentConfig,
    Pooling,
    Provider,
    TokenizerConfig,
    config_hash,
    load_datasets,
    load_experiment,
)


def test_shipped_configs_validate():
    experiment, datasets = load_experiment(), load_datasets()
    assert experiment.model("gemini-embedding-001").card_verified
    assert datasets.dataset("aila_statutes").tokenizer.tokenizer == "word_v4"


def test_only_deliberately_verified_models_can_encode():
    """card_verified means someone transcribed the prompting and checked it. Count, not list:
    the roster changes as the board grows, but an accidental flip should still be caught."""
    experiment = load_experiment()
    verified = [m.id for m in experiment.embedding_models if m.card_verified]
    unverified = [m.id for m in experiment.embedding_models if not m.card_verified]
    assert len(verified) == 23, f"unexpected verified roster: {verified}"
    assert unverified == ["nemotron-3-embed-8b"], unverified


def test_quantized_rows_share_an_endpoint_but_are_distinct_models():
    """RTEB lists embed-v4.0 at both 1536/float32 and 512/int8 -- same API, different models.

    `id` must stay unique (namespaces, result files, cache keys) while `model_name` is what the
    provider sees.
    """
    experiment = load_experiment()
    full = experiment.model("voyage-4-large")
    quant = experiment.model("voyage-3.5-int8-512")
    assert quant.model_name == "voyage-3.5" and quant.id != quant.model_name
    assert (quant.embd_dtype, quant.dim) == ("int8", 512)

    from src.index.embed import get_embedder
    # Routing follows model_name, not id: no vendor would recognise "voyage-3.5-int8-512".
    assert type(get_embedder(quant)) is type(get_embedder(full))


def test_voyage_law_2_is_not_given_the_family_prefix():
    """RTEB records the Represent-the-query prefix for the Voyage family but NOT for law-2.

    Voyage's input_type *is* that prefix, so law-2 must go out with none. Same vendor, same
    client, opposite prompting -- and getting it wrong produces no error.
    """
    experiment = load_experiment()
    law = experiment.model("voyage-law-2")
    assert (law.query_task_type, law.doc_task_type) == (None, None)
    for sibling in ("voyage-4", "voyage-4-lite", "voyage-3-large", "voyage-4-large"):
        assert experiment.model(sibling).query_task_type == "query", sibling


def test_voyage_does_not_double_its_own_prefix():
    """input_type makes Voyage prepend internally; adding the same string would duplicate it."""
    voyage = load_experiment().model("voyage-4-large")
    assert (voyage.query_task_type, voyage.doc_task_type) == ("query", "document")
    assert voyage.query_instruction is None and voyage.doc_instruction is None


def test_output_dtype_is_part_of_the_cache_key():
    """§9: dtype changes the vectors, so a bump must invalidate cached embeddings.

    RTEB treats quantized output as a distinct leaderboard row for the same model, which is the
    clearest evidence that dtype is model identity rather than a storage detail.
    """
    from src.index.embed import get_embedder

    voyage = load_experiment().model("voyage-4-large")
    quantized = voyage.model_copy(update={"embd_dtype": "int8"})

    key = get_embedder(voyage).cache_key("a statute", "doc")
    assert key != get_embedder(quantized).cache_key("a statute", "doc")


def test_batch_size_of_one_is_rejected():
    """voyage-4-large returns a different vector for a one-item request than for a batch."""
    with pytest.raises(ValidationError):
        load_experiment().model("voyage-4-large").model_copy(
            update={"batch_size": 1}
        ).model_validate({"id": "x", "provider": "api", "family": "api", "dim": 8, "batch_size": 1})


def test_batch_size_is_part_of_the_cache_key():
    """Batch size changes the vectors for some providers, so it is a version field (§9)."""
    from src.index.embed import get_embedder

    voyage = load_experiment().model("voyage-4-large")
    rebatched = voyage.model_copy(update={"batch_size": 32})
    assert get_embedder(voyage).cache_key("s", "doc") != get_embedder(rebatched).cache_key(
        "s", "doc"
    )


def test_cache_key_separates_queries_from_documents():
    """The same string embedded as a query and as a doc is two different vectors."""
    from src.index.embed import get_embedder

    embedder = get_embedder(load_experiment().model("voyage-4-large"))
    assert embedder.cache_key("text", "query") != embedder.cache_key("text", "doc")


def test_hosted_api_model_may_carry_a_text_instruction():
    """RTEB prefixes voyage-4-large but not gemini-embedding-001, so this must stay expressible.

    Assuming "hosted API implies no prefix" would drop Voyage's instruction silently -- exactly
    the §10 failure mode where a missing prefix costs several NDCG points with no error.
    """
    voyage = EmbeddingModelConfig(
        id="voyage-4-large", provider=Provider.API, family="api", dim=2048,
        query_instruction="Represent the query for retrieving supporting documents: ",
        doc_instruction="Represent the document for retrieval: ",
        query_task_type="query", doc_task_type="document",
    )
    assert voyage.query_instruction.endswith(": ")


def test_pooling_is_rejected_wherever_we_do_not_run_the_forward_pass():
    """Novita serves open weights but pools server-side, so pooling there would be inert."""
    for provider in (Provider.API, Provider.VERTEX, Provider.NOVITA):
        with pytest.raises(ValidationError, match="control of the forward pass"):
            EmbeddingModelConfig(
                id="x", provider=provider, family="f", dim=8, pooling=Pooling.MEAN,
            )


def test_open_weight_model_rejects_task_type():
    for provider in (Provider.NOVITA, Provider.LOCAL):
        with pytest.raises(ValidationError, match="hosted-API concept"):
            EmbeddingModelConfig(
                id="qwen3-embedding-8b", provider=provider, family="llm-backbone", dim=4096,
                query_task_type="RETRIEVAL_QUERY",
            )


def test_verified_local_model_requires_pooling():
    """Only the local route controls pooling, so only it can be wrong about it."""
    with pytest.raises(ValidationError, match="explicit pooling"):
        EmbeddingModelConfig(
            id="BAAI/bge-base-en-v1.5", provider=Provider.LOCAL, family="bi-encoder", dim=768,
            card_verified=True,
        )


def test_rteb_runs_open_weight_models_without_their_card_instructions():
    """Measured, not assumed: bare reproduces the published numbers, instructed does not.

    bge-base's card prefix raises AILAStatutes by 0.022 and HumanEval by 0.027 -- both far
    outside tolerance -- while bare lands within 0.0001 of published on both. Qwen3 likewise.
    The leaderboard therefore understates these models, and reproducing it means running bare.
    """
    experiment = load_experiment()
    for model_id in ("BAAI/bge-base-en-v1.5", "qwen/qwen3-embedding-8b"):
        model = experiment.model(model_id)
        assert model.query_instruction is None, model_id


def test_tokenizer_must_be_named_explicitly():
    """§9: `word_v4` is written out so an upstream default change cannot move BM25 silently."""
    with pytest.raises(ValidationError):
        TokenizerConfig(stemming=True, remove_stopwords=True)


def test_duplicate_model_ids_rejected():
    model = {"id": "dup", "provider": "api", "family": "api", "dim": 8}
    with pytest.raises(ValidationError, match="duplicate ids"):
        ExperimentConfig(embedding_models=[model, model], retrieval_modes=["dense"])


def test_unknown_lookups_name_the_alternatives():
    with pytest.raises(KeyError, match="gemini-embedding-001"):
        load_experiment().model("no-such-model")
    with pytest.raises(KeyError, match="aila_statutes"):
        load_datasets().dataset("no-such-dataset")


def test_typos_are_not_silently_absorbed():
    with pytest.raises(ValidationError):
        ExperimentConfig(
            embedding_models=[], retrieval_modes=["dense"], temprature=0.7  # noqa: F821 (typo)
        )


def test_config_hash_tracks_content_not_object_identity():
    experiment = load_experiment()
    other = ExperimentConfig.model_validate(
        experiment.model_dump(mode="json") | {"seed": experiment.seed + 1}
    )
    assert config_hash(experiment) == config_hash(experiment)
    assert config_hash(experiment) != config_hash(other)


def test_dataset_counts_are_the_pinned_revisions_not_the_card_prose():
    """The README says 197 statutes; the pinned corpus.jsonl ships 82. Config records reality."""
    aila = DatasetsConfig.model_validate(
        {"datasets": {"aila_statutes": load_datasets().dataset("aila_statutes").model_dump()}}
    ).dataset("aila_statutes")
    assert (aila.n_queries, aila.n_docs, aila.n_qrels) == (50, 82, 217)
    assert aila.hf_revision and aila.qrels_hf_revision, "revisions must be pinned (§9)"


def test_reproduction_target_is_recorded_for_the_phase_0_model():
    aila = load_datasets().dataset("aila_statutes")
    assert aila.published_ndcg_at_10["gemini-embedding-001"] == 0.45695
