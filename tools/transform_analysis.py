"""Stage 3: the transform-phase analysis — the two capstones.

Per dataset, computes Kendall tau + RBO between the human-baseline leaderboard and every
(condition x generator) transformed leaderboard, over the board models. For TREC-COVID adds a
seeded n=50 bootstrap CI on tau. Then:
  Capstone A (CUREv1 metric-blindness): is verbose tau ~0.95+ (harmless broadening / robustness, as
    ChatDoctor's 1-gold showed) or lower (metric blindness — CUREv1's ~40 dense golds see the harm)?
  Capstone B (TREC-COVID Q2 validation): does the SYNTHETIC transform reproduce the HUMAN phrasing
    shift, model by model? Spearman between human (query-question) and synthetic (terse-question)
    dNDCG vectors, and human (narrative-question) vs synthetic (verbose-question).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_experiment
from src.eval.correlation import kendall_tau, rbo, ranking

RESULTS = ROOT / "results"
GENS = ["gemini-2.5-pro", "claude-opus-4-8", "gpt-5.1-2025-11-13", "meta-llama/llama-3.3-70b-instruct"]
CONDS = ["paraphrase", "terse", "verbose"]
RNG = np.random.default_rng(20260720)


def flat(x: str) -> str:
    return x.replace("/", "__")


def per_query(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d.get("per_query") or None


def board_models(ds: str) -> list[str]:
    ms = [m.id for m in load_experiment().embedding_models if "Octen" not in m.id]
    return [m for m in ms if (RESULTS / f"dense__{ds}__{flat(m)}.json").exists()]


def mean_lb(pq_by_model: dict[str, dict[str, float]], qids=None) -> dict[str, float]:
    return {m: float(np.mean([pq[q] for q in (qids or pq)])) for m, pq in pq_by_model.items()}


def boot_tau_ci(base_pq, trans_pq, qids, n=1000):
    ts = []
    for _ in range(n):
        s = list(RNG.choice(qids, size=len(qids), replace=True))
        a = mean_lb(base_pq, s)
        b = mean_lb(trans_pq, s)
        ts.append(kendall_tau(a, b)[0])
    return np.percentile(ts, [2.5, 97.5])


DATASETS = sys.argv[1:] or ["chatdoctor", "curev1_en", "trec_covid"]

for ds in DATASETS:
    models = board_models(ds)
    base_pq = {m: per_query(RESULTS / f"dense__{ds}__{flat(m)}.json") for m in models}
    base_pq = {m: pq for m, pq in base_pq.items() if pq}
    qids = sorted(next(iter(base_pq.values())))
    base_lb = mean_lb(base_pq)
    print(f"\n{'='*74}\n{ds}  ({len(base_pq)} models, {len(qids)} queries)\n{'='*74}")
    print(f"{'condition':11s} {'generator':16s} {'tau':>8s} {'RBO':>7s}"
          + (f" {'95% CI (n=50 bootstrap)':>26s}" if ds == "trec_covid" else ""))
    tau_by_cond = {c: [] for c in CONDS}
    for cond in CONDS:
        for gen in GENS:
            tpq = {m: per_query(RESULTS / f"dense__{ds}__{flat(m)}__{cond}__{flat(gen)}.json")
                   for m in base_pq}
            tpq = {m: pq for m, pq in tpq.items() if pq}
            common = [m for m in base_pq if m in tpq]
            b = {m: base_lb[m] for m in common}
            t = mean_lb({m: tpq[m] for m in common})
            tau, _ = kendall_tau(b, t)
            r = rbo(ranking(b), ranking(t), 0.9)
            tau_by_cond[cond].append(tau)
            ci = ""
            if ds == "trec_covid":
                lo, hi = boot_tau_ci({m: base_pq[m] for m in common},
                                     {m: tpq[m] for m in common}, qids)
                ci = f"  [{lo:+.3f}, {hi:+.3f}]"
            print(f"{cond:11s} {gen.split('/')[-1]:16s} {tau:+8.4f} {r:7.4f}{ci}")
    print("\n  tau range by condition (across 4 generators):")
    for c in CONDS:
        v = tau_by_cond[c]
        print(f"    {c:11s} {min(v):+.4f} .. {max(v):+.4f}")

# ---------------- Capstone B: TREC-COVID human vs synthetic (Q2) ----------------
print(f"\n{'='*74}\nCAPSTONE B — TREC-COVID: do synthetic transforms proxy the human phrasing shift?\n{'='*74}")
ds = "trec_covid"
models = board_models(ds)
base = {m: per_query(RESULTS / f"dense__{ds}__{flat(m)}.json") for m in models}
base = {m: (np.mean(list(pq.values())) if pq else None) for m, pq in base.items()}

def field_mean(m, field):  # human query/narrative field result
    pq = per_query(RESULTS / f"dense__{ds}__{flat(m)}__{field}__human.json")
    return np.mean(list(pq.values())) if pq else None

def syn_mean(m, cond):  # synthetic condition, avg over generators
    vs = [np.mean(list(per_query(RESULTS / f"dense__{ds}__{flat(m)}__{cond}__{flat(g)}.json").values()))
          for g in GENS if per_query(RESULTS / f"dense__{ds}__{flat(m)}__{cond}__{flat(g)}.json")]
    return float(np.mean(vs)) if vs else None

pairs = [("human query - question", "synthetic terse - question", "query", "terse"),
         ("human narrative - question", "synthetic verbose - question", "narrative", "verbose")]
for hlabel, slabel, field, cond in pairs:
    hd, sd = [], []
    for m in models:
        if base.get(m) is None:
            continue
        hf, sc = field_mean(m, field), syn_mean(m, cond)
        if hf is None or sc is None:
            continue
        hd.append(hf - base[m])
        sd.append(sc - base[m])
    rho, p = stats.spearmanr(hd, sd)
    sign = np.mean(np.sign(hd) == np.sign(sd))
    print(f"  {hlabel:26s} vs {slabel:28s}  Spearman={rho:+.3f} (p={p:.3f}, n={len(hd)})  sign-agree={sign:.0%}")


# ---------------- Lexical control: BM25 isolates surface-word change ----------------
def bm25_mean(ds: str, cond: str | None = None, gen: str | None = None) -> float | None:
    if cond is None:
        hits = [p for p in RESULTS.glob(f"bm25__{ds}__*.json") if p.stem.count("__") == 2]
    else:
        hits = list(RESULTS.glob(f"bm25__{ds}__*__{cond}__{flat(gen)}.json"))
    pq = per_query(hits[0]) if hits else None
    return float(np.mean(list(pq.values()))) if pq else None


print(f"\n{'='*74}\nLEXICAL CONTROL — BM25 dNDCG@10 against the human-query BM25 run\n{'='*74}")
print(f"{'dataset':12s} {'baseline':>9s}" + "".join(f"{c:>12s}" for c in CONDS))
for ds in DATASETS:
    bm_base = bm25_mean(ds)
    if bm_base is None:
        continue
    cells = []
    for cond in CONDS:
        vs = [v for v in (bm25_mean(ds, cond, g) for g in GENS) if v is not None]
        cells.append(f"{np.mean(vs) - bm_base:+12.3f}" if vs else f"{'n/a':>12s}")
    print(f"{ds:12s} {bm_base:9.4f}" + "".join(cells))

# ---------------- Levelling: does elaboration cost the strongest models most? ----------------
# Correlating a change against the baseline it was computed from invites regression to the mean,
# so strength is also measured on a SEPARATE dataset and both readings are printed.
def ds_mean(ds: str, m: str, suffix: str = "") -> float | None:
    pq = per_query(RESULTS / f"dense__{ds}__{flat(m)}{suffix}.json")
    return float(np.mean(list(pq.values()))) if pq else None


def shifted_mean(ds: str, m: str, cond: str | None, field: str | None) -> float | None:
    if field:
        return ds_mean(ds, m, f"__{field}__human")
    vs = [v for v in (ds_mean(ds, m, f"__{cond}__{flat(g)}") for g in GENS) if v is not None]
    return float(np.mean(vs)) if vs else None


LEVELLING = [
    ("trec_covid", "verbose", None, "curev1_en", "TREC-COVID verbose"),
    ("curev1_en", "verbose", None, "trec_covid", "CUREv1 verbose"),
    ("trec_covid", None, "narrative", "curev1_en", "TREC-COVID human narrative"),
]

print(f"\n{'='*74}\nLEVELLING — model strength vs the dNDCG@10 the shift costs it\n{'='*74}")
print(f"{'shift':28s} {'strength basis':24s} {'rho':>7s} {'p':>8s} {'n':>4s}")
for ds, cond, field, other, label in LEVELLING:
    deltas, indep, naive = [], [], []
    for m in board_models(ds):
        base, after, elsewhere = ds_mean(ds, m), shifted_mean(ds, m, cond, field), ds_mean(other, m)
        if base is None or after is None or elsewhere is None:
            continue
        deltas.append(after - base)
        indep.append(elsewhere)
        naive.append(base)
    if not deltas:
        continue
    for basis, strength in ((f"{other} (independent)", indep), ("own baseline (naive)", naive)):
        rho, p = stats.spearmanr(strength, deltas)
        print(f"{label:28s} {basis:24s} {rho:+7.3f} {p:8.4f} {len(deltas):4d}")
