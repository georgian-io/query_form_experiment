"""Why did ΔNDCG move? Per-query attribution, and a test of the memorization hypothesis (§5.5).

Two questions, run over per-query scores already on disk.

**Attribution.** §5.5 asks that ΔNDCG be regressed on query features. The one that matters here
is lexical: how much of the original wording survived the rewrite. If a model's loss tracks
lexical change steeply, it was relying on the surface form rather than the meaning.

**The memorization test.** `qwen3-embedding-8b` scores 0.807 on human AILAStatutes queries
against ~0.46 for every other model, and is the only model that collapses under paraphrase. If
that advantage came from having seen these verbatim Indian Supreme Court excerpts in
pretraining, then the queries where it most out-performs the field should be exactly the queries
it loses under paraphrase -- the advantage and its destruction should be the same phenomenon.

The test is `corr(excess_over_field_on_human, per_query_delta)`. Strongly negative means the
advantage is phrasing-specific. Run against every model, because a mildly negative correlation
is expected everywhere (a model that scores well has more room to fall); the claim needs Qwen3
to be an outlier, not merely negative.

    uv run python scripts/05b_attribution.py --dataset aila_statutes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy import stats  # noqa: E402

from src.config import load_datasets  # noqa: E402
from src.data import load_dataset  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--suspect", default="qwen/qwen3-embedding-8b")
    args = parser.parse_args()

    dc = load_datasets().dataset(args.dataset)
    ds = load_dataset(args.dataset, dc)
    originals, _corpus, _qrels = ds.queries(), ds.corpus(), ds.qrels()

    human, trans = {}, defaultdict(dict)
    for p in RESULTS_DIR.glob(f"dense__{args.dataset}__*.json"):
        r = json.loads(p.read_text())
        if r["condition"] == "human":
            human[r["model"]] = r["per_query"]
        else:
            trans[r["generator"]][r["model"]] = r["per_query"]

    rewrites = {}
    for p in RESULTS_DIR.glob(f"queries__{args.dataset}__paraphrase__*.json"):
        r = json.loads(p.read_text())
        rewrites[r["generator"]] = r["queries"]

    models = sorted(set(human) & set.intersection(*(set(v) for v in trans.values())))
    generators = sorted(trans)
    print(f"{args.dataset}: {len(models)} models x {len(generators)} generators\n")

    # ---- 1. Does the loss track how much wording changed? -------------------
    print("=" * 92)
    print("LEXICAL ATTRIBUTION — corr(query-vs-rewrite overlap, per-query ΔNDCG)")
    print("negative = the model loses more where the wording changed more (surface-form reliance)")
    print("=" * 92)
    print(f"{'model':<40}" + "".join(f"{g.split('/')[-1][:10]:>12}" for g in generators))
    for m in models:
        cells = []
        for g in generators:
            qids = [q for q in originals if q in rewrites[g] and q in human[m]]
            overlap = [jaccard(originals[q], rewrites[g][q]) for q in qids]
            delta = [trans[g][m][q] - human[m][q] for q in qids]
            r = stats.pearsonr(overlap, delta)
            cells.append(f"{r.statistic:>+9.2f}{'*' if r.pvalue < 0.05 else ' '} ")
        print(f"{m:<40}" + "".join(cells))

    # ---- 2. Is the suspect's advantage the thing paraphrase destroys? -------
    print("\n" + "=" * 92)
    print("MEMORIZATION TEST — corr(excess over field on HUMAN queries, per-query ΔNDCG)")
    print("strongly negative = the model's edge lives in the exact wording, and is destroyed by")
    print("restating it. Expected mildly negative for all (headroom); the claim needs an outlier.")
    print("=" * 92)
    print(f"{'model':<40}{'human':>8}{'excess':>9}" +
          "".join(f"{g.split('/')[-1][:10]:>12}" for g in generators))
    field = {}
    for q in originals:
        vals = [human[m][q] for m in models if q in human[m]]
        field[q] = sum(vals) / len(vals) if vals else 0.0

    for m in models:
        qids = [q for q in originals if q in human[m]]
        excess = [human[m][q] - field[q] for q in qids]
        mean_h = sum(human[m][q] for q in qids) / len(qids)
        cells = []
        for g in generators:
            delta = [trans[g][m][q] - human[m][q] for q in qids]
            r = stats.pearsonr(excess, delta)
            cells.append(f"{r.statistic:>+9.2f}{'*' if r.pvalue < 0.05 else ' '} ")
        flag = "  <-- suspect" if m == args.suspect else ""
        print(f"{m:<40}{mean_h:>8.4f}{sum(excess) / len(excess):>+9.4f}"
              + "".join(cells) + flag)

    # ---- 3. Where does the suspect's advantage actually live? ---------------
    if args.suspect in models:
        print("\n" + "=" * 92)
        print(f"WHERE {args.suspect} WINS, AND WHETHER IT KEEPS WINNING THERE")
        print("=" * 92)
        qids = sorted(originals, key=lambda q: -(human[args.suspect][q] - field[q]))
        top = qids[: len(qids) // 3]
        rest = qids[len(qids) // 3:]
        for label, group in (("top third by excess", top), ("other two thirds", rest)):
            h = sum(human[args.suspect][q] for q in group) / len(group)
            f = sum(field[q] for q in group) / len(group)
            deltas = [
                sum(trans[g][args.suspect][q] - human[args.suspect][q] for q in group) / len(group)
                for g in generators
            ]
            print(f"  {label:<22} n={len(group):<4} qwen human {h:.3f}  field {f:.3f}  "
                  f"edge {h - f:+.3f}   mean Δ {sum(deltas) / len(deltas):+.4f}")
        print("\n  If the edge and the loss concentrate in the same queries, the advantage is")
        print("  phrasing-specific -- consistent with having memorized the source text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
