"""Loader for RTEB's packaged dataset layout.

RTEB ships every dataset as three flat jsonl files -- `queries.jsonl`, `corpus.jsonl`,
`relevance.jsonl` -- using BEIR-style field names and `id` where mteb uses `_id`. That shape is
identical across datasets, so it lives here and per-dataset adapters carry only genuine quirks
(§5.1). A subclass that adds nothing but a name is the correct outcome, not a smell: it means
the dataset had no quirks to absorb.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.data.base import Corpus, Dataset, Document, Qrels, Queries


class RTEBJsonlDataset(Dataset):
    """Standard RTEB three-file packaging, with optional cross-repo qrels.

    `qrels_hf_repo` exists because RTEB's published relevance files are not always complete --
    AILAStatutes ships one gold per query where the leaderboard was scored against all of them.
    When qrels are borrowed, the two repos are checked to agree on the *task* before trusting it.
    """

    def _read_jsonl(
        self, filename: str, repo: str | None = None, revision: str | None = None
    ) -> list[dict]:
        path = Path(
            hf_hub_download(
                repo_id=repo or self.config.hf_repo,
                filename=filename,
                revision=revision if repo else self.config.hf_revision,
                repo_type="dataset",
            )
        )
        with path.open() as fh:
            return [json.loads(line) for line in fh if line.strip()]

    @staticmethod
    def _row_id(row: dict) -> str:
        """RTEB is not consistent about the id field across its own datasets.

        AILAStatutes and HumanEval use `id`; ChatDoctor_HealthCareMagic uses mteb's `_id`.
        Accept either rather than adding a per-dataset adapter for a one-character difference,
        but fail loudly if neither is present -- silently inventing ids would misalign the qrels.
        """
        for key in ("id", "_id"):
            if key in row:
                return row[key]
        raise KeyError(f"row has neither 'id' nor '_id': {sorted(row)[:5]}")

    def _load_queries(self) -> Queries:
        return {self._row_id(r): r["text"] for r in self._read_jsonl("queries.jsonl")}

    def _load_corpus(self) -> Corpus:
        # `title` is absent entirely in some packagings and empty in others; RTEB's own loader
        # reads `text` only, so a missing title is normal rather than a defect.
        return {
            self._row_id(r): Document(title=r.get("title", ""), text=r["text"])
            for r in self._read_jsonl("corpus.jsonl")
        }

    def _load_qrels(self) -> Qrels:
        repo = self.config.qrels_hf_repo
        if repo is None:
            rows = self._read_jsonl("relevance.jsonl")
        else:
            revision = self.config.qrels_hf_revision
            self._assert_repos_agree(repo, revision)
            rows = self._read_jsonl(self.config.qrels_path, repo=repo, revision=revision)

        qrels: Qrels = {}
        for row in rows:
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        return qrels

    def _assert_repos_agree(self, repo: str, revision: str | None) -> None:
        """Borrowing qrels across repos is only sound if the other repo is the same task.

        Compares ids *and* text, not just counts: two repos could agree on every id while
        differing in document content, which would silently change what the leaderboard number
        means.
        """
        for filename, label in (("corpus.jsonl", "corpus"), ("queries.jsonl", "queries")):
            ours = {self._row_id(r): r["text"] for r in self._read_jsonl(filename)}
            theirs = {
                self._row_id(r): r["text"]
                for r in self._read_jsonl(filename, repo, revision)
            }
            if ours != theirs:
                raise ValueError(
                    f"{self.name}: {label} differ between {self.config.hf_repo} and {repo}; "
                    "qrels cannot be borrowed across repos that disagree on the task"
                )
