"""BM25 control tests (§5.4).

The one behaviour worth pinning without a live index: a query over turbopuffer's 8192-code-point
BM25 limit is truncated, not passed through to a 400 that would kill the whole lexical run.
"""

from __future__ import annotations

from src.retrieve.bm25 import _BM25_MAX_CODEPOINTS, bm25, bm25_batch


class _RecordingIndex:
    """Stub Index that records the query text it was handed."""

    def __init__(self):
        self.seen: list[str] = []

    def bm25(self, query_text, top_k=10):
        self.seen.append(query_text)
        return [("d1", 1.0)]


def test_overlong_query_is_truncated_to_the_tpuf_limit():
    idx = _RecordingIndex()
    bm25(idx, "x" * 20000)
    assert len(idx.seen[0]) == _BM25_MAX_CODEPOINTS


def test_normal_query_passes_through_unchanged():
    idx = _RecordingIndex()
    bm25(idx, "chest pain three days")
    assert idx.seen[0] == "chest pain three days"


def test_batch_truncates_each_query_independently():
    idx = _RecordingIndex()
    bm25_batch(idx, ["a", "b"], {"a": "y" * 9000, "b": "short"})
    assert len(idx.seen[0]) == _BM25_MAX_CODEPOINTS
    assert idx.seen[1] == "short"
