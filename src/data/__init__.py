"""Dataset adapter registry.

Keyed by the `adapter` field in datasets.yaml so a run is selected entirely from config.
"""

from __future__ import annotations

from src.config import DatasetConfig, load_datasets
from src.data.aila import AILAStatutes
from src.data.base import Corpus, Dataset, Document, Qrels, Queries
from src.data.chatdoctor import ChatDoctorHealthCareMagic
from src.data.curev1 import CUREv1En
from src.data.humaneval import HumanEval
from src.data.trec_covid import TrecCovid

ADAPTERS: dict[str, type[Dataset]] = {
    "aila": AILAStatutes,
    "humaneval": HumanEval,
    "chatdoctor": ChatDoctorHealthCareMagic,
    "curev1": CUREv1En,
    "trec_covid": TrecCovid,
}


def load_dataset(name: str, config: DatasetConfig | None = None) -> Dataset:
    cfg = config or load_datasets().dataset(name)
    if cfg.adapter not in ADAPTERS:
        raise KeyError(f"no adapter registered for {cfg.adapter!r}; have: {', '.join(ADAPTERS)}")
    return ADAPTERS[cfg.adapter](cfg)


__all__ = [
    "ADAPTERS",
    "Corpus",
    "Dataset",
    "Document",
    "Qrels",
    "Queries",
    "load_dataset",
]
