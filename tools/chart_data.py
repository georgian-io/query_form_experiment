"""Extract the numbers behind the blog-post graphics into one JSON blob.

Emits, for the three query-transform datasets:
  human_fields  - TREC-COVID per-model mean NDCG@10 under each human phrasing (slopegraph)
  tau_grid      - tau per (dataset, condition, generator) vs the human-query board
  deltas        - per-model mean dNDCG@10 per condition, averaged over generators (heatmap)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.correlation import kendall_tau  # noqa: E402

RESULTS = ROOT / "results"
GENS = [
    "gemini-2.5-pro",
    "claude-opus-4-8",
    "gpt-5.1-2025-11-13",
    "meta-llama/llama-3.3-70b-instruct",
]
CONDS = ["paraphrase", "terse", "verbose"]
DATASETS = ["chatdoctor", "curev1_en", "trec_covid"]
DENSITY = {"chatdoctor": 1.0, "curev1_en": 40.0, "trec_covid": 493.6}


def flat(x: str) -> str:
    return x.replace("/", "__")


def per_query(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("per_query") or None


def mean_of(pq: dict[str, float] | None) -> float | None:
    return float(np.mean(list(pq.values()))) if pq else None


def board_models(ds: str) -> list[str]:
    paths = RESULTS.glob(f"dense__{ds}__*.json")
    out = []
    for p in paths:
        tail = p.stem[len(f"dense__{ds}__") :]
        if any(k in tail for k in CONDS) or "__human" in tail:
            continue
        out.append(tail)
    return sorted(set(out))


def pretty(model_flat: str) -> str:
    return model_flat.replace("__", "/")


data: dict = {"tau_grid": [], "deltas": {}, "human_fields": []}

# ---- tau grid: every (dataset, condition, generator) ------------------------
for ds in DATASETS:
    models = board_models(ds)
    base_pq = {m: per_query(RESULTS / f"dense__{ds}__{m}.json") for m in models}
    base_pq = {m: pq for m, pq in base_pq.items() if pq}
    base_lb = {m: mean_of(pq) for m, pq in base_pq.items()}
    for cond in CONDS:
        for gen in GENS:
            tpq = {
                m: per_query(RESULTS / f"dense__{ds}__{m}__{cond}__{flat(gen)}.json")
                for m in base_pq
            }
            tpq = {m: pq for m, pq in tpq.items() if pq}
            common = [m for m in base_pq if m in tpq]
            if len(common) < 3:
                continue
            b = {m: base_lb[m] for m in common}
            t = {m: mean_of(tpq[m]) for m in common}
            tau, _ = kendall_tau(b, t)
            data["tau_grid"].append(
                {
                    "dataset": ds,
                    "density": DENSITY[ds],
                    "condition": cond,
                    "generator": gen.split("/")[-1],
                    "tau": round(float(tau), 4),
                    "n_models": len(common),
                }
            )

# ---- per-model deltas, averaged across generators ---------------------------
for ds in ["curev1_en", "trec_covid"]:
    models = board_models(ds)
    rows = []
    for m in models:
        base = mean_of(per_query(RESULTS / f"dense__{ds}__{m}.json"))
        if base is None:
            continue
        row = {"model": pretty(m), "baseline": round(base, 4)}
        for cond in CONDS:
            vals = [
                mean_of(per_query(RESULTS / f"dense__{ds}__{m}__{cond}__{flat(g)}.json"))
                for g in GENS
            ]
            vals = [v for v in vals if v is not None]
            row[cond] = round(float(np.mean(vals)) - base, 4) if vals else None
        rows.append(row)
    rows.sort(key=lambda r: -r["baseline"])
    data["deltas"][ds] = rows

# ---- TREC-COVID human phrasing fields (slopegraph) -------------------------
for m in board_models("trec_covid"):
    question = mean_of(per_query(RESULTS / f"dense__trec_covid__{m}.json"))
    keyword = mean_of(per_query(RESULTS / f"dense__trec_covid__{m}__query__human.json"))
    narrative = mean_of(per_query(RESULTS / f"dense__trec_covid__{m}__narrative__human.json"))
    if None in (question, keyword, narrative):
        continue
    data["human_fields"].append(
        {
            "model": pretty(m),
            "keyword": round(keyword, 4),
            "question": round(question, 4),
            "narrative": round(narrative, 4),
        }
    )

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("chart_data.json")
out.write_text(json.dumps(data, indent=1))

print(f"tau_grid rows : {len(data['tau_grid'])}")
for ds in DATASETS:
    rows = [r for r in data["tau_grid"] if r["dataset"] == ds]
    if rows:
        n = rows[0]["n_models"]
        for cond in CONDS:
            taus = [r["tau"] for r in rows if r["condition"] == cond]
            print(f"  {ds:12s} {cond:11s} n={n:3d}  tau {min(taus):+.3f}..{max(taus):+.3f}")
print(f"human_fields  : {len(data['human_fields'])} models")
for ds, rows in data["deltas"].items():
    print(f"deltas {ds:12s}: {len(rows)} models")
print(f"wrote {out}")
