"""Build the four candidate blog graphics as one self-contained HTML review page.

Reads the JSON emitted by chart_data.py; writes an HTML file with inline SVG.
Palette + mark specs follow the validated reference instance of the dataviz skill.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DATA = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2])

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (light)
GRID, AXIS, MUTED = "var(--grid)", "var(--axis)", "var(--muted)"

# diverging ramp for the heatmap: red = hurts, blue = helps, gray midpoint
RED = ["#a52a2a", "#c8413f", "#de6b66", "#eea19c", "#f8d5d1"]
BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#17518f"]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def short(m: str) -> str:
    m = m.replace("sentence-transformers/", "").replace("BAAI/", "").replace("qwen/", "")
    return m if len(m) <= 26 else m[:25] + "…"


def lerp_hex(a: str, b: str, t: float) -> str:
    ar, ag, ab = (int(a[i : i + 2], 16) for i in (1, 3, 5))
    br, bg, bb = (int(b[i : i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (
        round(ar + (br - ar) * t),
        round(ag + (bg - ag) * t),
        round(ab + (bb - ab) * t),
    )


def diverging(v: float, vmax: float) -> str:
    """Map a delta to the diverging ramp. Negative = red (hurt), positive = blue (help)."""
    if v is None:
        return "var(--surface-2)"
    t = max(-1.0, min(1.0, v / vmax))
    if abs(t) < 0.04:
        return "var(--mid)"
    ramp = BLUE[::-1] if t > 0 else RED[::-1]
    # ramp[0] = palest .. ramp[-1] = deepest
    pos = abs(t) * (len(ramp) - 1)
    i = min(int(pos), len(ramp) - 2)
    return lerp_hex(ramp[i], ramp[i + 1], pos - i)


def ink_for(fill: str) -> str:
    if fill.startswith("var"):
        return "var(--text-secondary)"
    r, g, b = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    return "#ffffff" if lum < 0.55 else "#0b0b0b"


figs: list[str] = []

# ---------------------------------------------------------------- FIG 1
hf = DATA["human_fields"]
FIELDS = [("keyword", "keyword"), ("question", "question"), ("narrative", "narrative")]
W, H = 830, 470
ML, MR, MT, MB = 100, 210, 26, 46
ranks = {
    k: {r["model"]: i + 1 for i, r in enumerate(sorted(hf, key=lambda r: -r[k]))}
    for k, _ in FIELDS
}
vals = [r[k] for r in hf for k, _ in FIELDS]
ymin, ymax = math.floor(min(vals) * 20) / 20, math.ceil(max(vals) * 20) / 20
xs = [ML + i * (W - ML - MR) / 2 for i in range(3)]


def y1(v: float) -> float:
    return MT + (H - MT - MB) * (1 - (v - ymin) / (ymax - ymin))


HL = {
    "text-embedding-3-large": S1,
    "BAAI/bge-large-en-v1.5": S2,
    "embed-multilingual-v3.0": S3,
}
p: list[str] = []
for gy in [round(ymin + i * 0.1, 2) for i in range(int((ymax - ymin) / 0.1) + 1)]:
    p.append(
        f'<line x1="{ML-8}" y1="{y1(gy):.1f}" x2="{W-MR+8}" y2="{y1(gy):.1f}" stroke="{GRID}" stroke-width="1"/>'
        f'<text x="{ML-16}" y="{y1(gy)+4:.1f}" text-anchor="end" class="tick">{gy:.1f}</text>'
    )
for i, (_, lab) in enumerate(FIELDS):
    p.append(f'<text x="{xs[i]:.0f}" y="{H-18}" text-anchor="middle" class="axlab">{lab}</text>')
for r in sorted(hf, key=lambda r: -r["question"]):
    m = r["model"]
    c = HL.get(m)
    pts = " ".join(f"{xs[i]:.1f},{y1(r[k]):.1f}" for i, (k, _) in enumerate(FIELDS))
    tip = (
        f"{esc(short(m))} &#183; keyword {r['keyword']:.3f} (#{ranks['keyword'][m]}) &#183; "
        f"question {r['question']:.3f} (#{ranks['question'][m]}) &#183; "
        f"narrative {r['narrative']:.3f} (#{ranks['narrative'][m]})"
    )
    cls = "hl" if c else "bg"
    p.append(
        f'<polyline class="ln {cls}" points="{pts}" fill="none" stroke="{c or "var(--faint)"}" '
        f'stroke-width="{2 if c else 1.4}" data-tip="{tip}"/>'
    )
    for i, (k, _) in enumerate(FIELDS):
        p.append(
            f'<circle class="dot {cls}" cx="{xs[i]:.1f}" cy="{y1(r[k]):.1f}" r="{4 if c else 3}" '
            f'fill="{c or "var(--faint)"}" stroke="var(--surface-1)" stroke-width="2" data-tip="{tip}"/>'
        )
    if c:
        p.append(
            f'<text x="{xs[2]+12:.0f}" y="{y1(r["narrative"])+4:.1f}" class="dlab" fill="{c}">{esc(short(m))}</text>'
        )
figs.append(
    {
        "n": 1,
        "title": "The same information need, three human phrasings",
        "sub": "TREC-COVID ships a keyword, a question, and a narrative for each of its 50 topics. "
        "No LLM touched these. Each line is one of 20 embedding models; crossings are the leaderboard reordering.",
        "svg": f'<svg viewBox="0 0 {W} {H}" class="chart">{"".join(p)}</svg>',
        "cap": "Kendall &tau; vs the <code>question</code> board: 0.821 to <code>keyword</code>, 0.726 to "
        "<code>narrative</code> &#8212; both below 0.9 with zero change in meaning. Note that "
        "<code>question</code> is the easiest form for almost every model, so the effect is not "
        "just &#8220;longer is harder.&#8221;",
        "place": "Fits inside &#167;1, right after the human-phrasing &tau; table.",
    }
)

# ---------------------------------------------------------------- FIG 2
grid = DATA["tau_grid"]
DS = [
    ("chatdoctor", "ChatDoctor", "1 gold", 17),
    ("curev1_en", "CUREv1", "~40 golds", 21),
    ("trec_covid", "TREC-COVID", "graded, ~494", 21),
]
CONDC = [("paraphrase", S1), ("terse", S3), ("verbose", S2)]
W2, H2 = 760, 430
ML2, MR2, MT2, MB2 = 92, 152, 26, 62
x2 = [ML2 + i * (W2 - ML2 - MR2) / 2 for i in range(3)]
lo, hi = 0.45, 1.0


def y2(v: float) -> float:
    return MT2 + (H2 - MT2 - MB2) * (1 - (v - lo) / (hi - lo))


q: list[str] = []
for gy in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    q.append(
        f'<line x1="{ML2-8}" y1="{y2(gy):.1f}" x2="{W2-MR2+8}" y2="{y2(gy):.1f}" stroke="{GRID}" stroke-width="1"/>'
        f'<text x="{ML2-16}" y="{y2(gy)+4:.1f}" text-anchor="end" class="tick">{gy:.1f}</text>'
    )
q.append(
    f'<line x1="{ML2-8}" y1="{y2(0.9):.1f}" x2="{W2-MR2+8}" y2="{y2(0.9):.1f}" stroke="var(--axis)" stroke-width="1.5" stroke-dasharray="1 0"/>'
    f'<text x="{W2-MR2+14}" y="{y2(0.9)-6:.1f}" class="note">&tau; = 0.9</text>'
)
for i, (_, nm, dens, nm_models) in enumerate(DS):
    q.append(f'<text x="{x2[i]:.0f}" y="{H2-30}" text-anchor="middle" class="axlab">{nm}</text>')
    q.append(f'<text x="{x2[i]:.0f}" y="{H2-14}" text-anchor="middle" class="sublab">{dens}</text>')
for cond, col in CONDC:
    means = []
    for i, (ds, _, _, _) in enumerate(DS):
        vs = [r["tau"] for r in grid if r["dataset"] == ds and r["condition"] == cond]
        means.append(sum(vs) / len(vs))
        for v in vs:
            q.append(
                f'<circle class="dot" cx="{x2[i]:.1f}" cy="{y2(v):.1f}" r="3.5" fill="{col}" '
                f'opacity="0.5" data-tip="{cond} &#183; {esc(nm)} &#183; &tau; {v:.3f} (one generator)"/>'
            )
    pts = " ".join(f"{x2[i]:.1f},{y2(m):.1f}" for i, m in enumerate(means))
    q.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
    for i, m in enumerate(means):
        q.append(
            f'<circle class="dot" cx="{x2[i]:.1f}" cy="{y2(m):.1f}" r="5" fill="{col}" '
            f'stroke="var(--surface-1)" stroke-width="2" data-tip="{cond} &#183; {esc(DS[i][1])} &#183; mean &tau; {m:.3f} over 4 generators"/>'
        )
    q.append(
        f'<text x="{x2[2]+14:.0f}" y="{y2(means[2])+4:.1f}" class="dlab" fill="{col}">{cond}</text>'
    )
figs.append(
    {
        "n": 2,
        "title": "The same rewrite, seen through three answer keys",
        "sub": "Kendall &tau; against each dataset's own human-query board. Small dots are the four "
        "LLM rewriters individually; the line joins their mean. Datasets are ordered by how many "
        "documents per query the benchmark actually labels.",
        "svg": f'<svg viewBox="0 0 {W2} {H2}" class="chart">{"".join(q)}</svg>',
        "cap": "Verbose elaboration looks like the <em>gentlest</em> move on a one-gold key and the "
        "<em>harshest</em> on dense keys &#8212; and all four rewriters agree. Paraphrase stays flat "
        "across all three, so this is not a story about every rewrite getting worse. "
        "<strong>Caveat for the caption:</strong> these are three different corpora, not one corpus "
        "re-labelled, and the ChatDoctor board carries 17 models against 21 for the others.",
        "place": "Anchors &#167;2 &#8212; replaces or sits beside the three-column &tau; table.",
    }
)

# ---------------------------------------------------------------- FIG 3
W3, H3 = 760, 330
q3: list[str] = []
PL, PT, PW, PH = 78, 46, 268, 200
q3.append(f'<text x="{PL}" y="26" class="panel">A &#183; what a one-gold key can see</text>')
for gy in [0, 0.25, 0.5, 0.75, 1.0]:
    yy = PT + PH * (1 - gy)
    q3.append(
        f'<line x1="{PL}" y1="{yy:.1f}" x2="{PL+PW}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>'
        f'<text x="{PL-12}" y="{yy+4:.1f}" text-anchor="end" class="tick">{gy:.2f}</text>'
    )
pts3 = []
for r in range(1, 11):
    v = 1 / math.log2(1 + r)
    xx = PL + PW * (r - 1) / 9
    yy = PT + PH * (1 - v)
    pts3.append(f"{xx:.1f},{yy:.1f}")
q3.append(f'<polyline points="{" ".join(pts3)}" fill="none" stroke="{S1}" stroke-width="2"/>')
for r in range(1, 11):
    v = 1 / math.log2(1 + r)
    xx = PL + PW * (r - 1) / 9
    yy = PT + PH * (1 - v)
    q3.append(
        f'<circle class="dot" cx="{xx:.1f}" cy="{yy:.1f}" r="4.5" fill="{S1}" stroke="var(--surface-1)" '
        f'stroke-width="2" data-tip="gold at rank {r} &#8594; NDCG@10 = {v:.3f}"/>'
    )
    if r in (1, 4, 10):
        q3.append(f'<text x="{xx:.0f}" y="{PT+PH+18}" text-anchor="middle" class="tick">{r}</text>')
q3.append(
    f'<text x="{PL+PW/2:.0f}" y="{PT+PH+38}" text-anchor="middle" class="axlab">rank of the single gold document</text>'
)
q3.append(f'<text x="{PL-58}" y="{PT-14}" class="sublab">NDCG@10</text>')
PL2 = 470
q3.append(f'<text x="{PL2}" y="26" class="panel">B &#183; two models, same gold at rank 1</text>')
BW = 210
for gy in [0, 0.25, 0.5, 0.75, 1.0]:
    yy = PT + PH * (1 - gy)
    q3.append(
        f'<line x1="{PL2}" y1="{yy:.1f}" x2="{PL2+BW}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>'
    )
AB = [("Model A", 1.000, 0.967, S1, "relevant at ranks 1, 2, 4"), ("Model B", 1.000, 0.469, S2, "relevant at rank 1 only")]
xk, xp = PL2 + 34, PL2 + BW - 34
for nm, sparse, pooled, col, det in AB:
    ys, yp = PT + PH * (1 - sparse), PT + PH * (1 - pooled)
    q3.append(
        f'<polyline points="{xk},{ys:.1f} {xp},{yp:.1f}" fill="none" stroke="{col}" stroke-width="2"/>'
    )
    for xx, vv in ((xk, sparse), (xp, pooled)):
        yy = PT + PH * (1 - vv)
        q3.append(
            f'<circle class="dot" cx="{xx}" cy="{yy:.1f}" r="5.5" fill="{col}" stroke="var(--surface-1)" '
            f'stroke-width="2" data-tip="{nm} ({det}) &#183; NDCG@10 {vv:.3f}"/>'
        )
    q3.append(f'<text x="{xp+12}" y="{yp+4:.1f}" class="dlab" fill="{col}">{nm}</text>')
    q3.append(f'<text x="{xp+12}" y="{yp+18:.1f}" class="sublab">{pooled:.3f}</text>')
q3.append(f'<text x="{xk}" y="{PT+PH+18}" text-anchor="middle" class="tick">one-gold key</text>')
q3.append(f'<text x="{xp}" y="{PT+PH+18}" text-anchor="middle" class="tick">pooled key</text>')
q3.append(
    f'<text x="{xk}" y="{PT+PH*(1-1.0)-14:.0f}" text-anchor="middle" class="note">both 1.000</text>'
)
figs.append(
    {
        "n": 3,
        "title": "Why a thin answer key cannot separate two models",
        "sub": "Left: with exactly one relevant document, NDCG@10 collapses to 1/log&#8322;(1+r) &#8212; a "
        "function of that document's rank and nothing else. Right: two models that place the gold "
        "identically and differ entirely in the nine slots the key never labelled.",
        "svg": f'<svg viewBox="0 0 {W3} {H3}" class="chart">{"".join(q3)}</svg>',
        "cap": "Judging the neighbours splits a gap the sparse key reported as exactly zero. This is "
        "arithmetic, not measurement &#8212; it holds for any one-gold benchmark.",
        "place": "Sits in &#167;2 immediately after the single-gold derivation.",
    }
)

# ---------------------------------------------------------------- FIG 4
rows = DATA["deltas"]["trec_covid"]
CONDS4 = ["paraphrase", "terse", "verbose"]
vmax = max(abs(r[c]) for r in rows for c in CONDS4 if r[c] is not None)
CW, CH, GAP = 122, 21, 3
W4 = 250 + 3 * (CW + GAP) + 118
H4 = 58 + len(rows) * (CH + GAP) + 22
q4: list[str] = []
for j, c in enumerate(CONDS4):
    q4.append(
        f'<text x="{250 + j*(CW+GAP) + CW/2:.0f}" y="44" text-anchor="middle" class="axlab">{c}</text>'
    )
q4.append('<text x="238" y="44" text-anchor="end" class="sublab">model (by baseline)</text>')
for i, r in enumerate(rows):
    yy = 58 + i * (CH + GAP)
    q4.append(
        f'<text x="238" y="{yy+CH/2+4:.0f}" text-anchor="end" class="rowlab">{esc(short(r["model"]))}</text>'
    )
    q4.append(
        f'<text x="{250 - 6:.0f}" y="{yy+CH/2+4:.0f}" text-anchor="end" class="tick" opacity="0"></text>'
    )
    for j, c in enumerate(CONDS4):
        v = r[c]
        xx = 250 + j * (CW + GAP)
        fill = diverging(v, vmax)
        q4.append(
            f'<rect class="cell" x="{xx}" y="{yy}" width="{CW}" height="{CH}" rx="3" fill="{fill}" '
            f'data-tip="{esc(short(r["model"]))} &#183; {c} &#183; &#916;NDCG@10 {v:+.3f} '
            f'(baseline {r["baseline"]:.3f} &#8594; {r["baseline"]+v:.3f})"/>'
        )
        if abs(v) >= 0.09:
            q4.append(
                f'<text x="{xx+CW/2:.0f}" y="{yy+CH/2+4:.0f}" text-anchor="middle" class="cellval" '
                f'fill="{ink_for(fill)}">{v:+.2f}</text>'
            )
lx = 250 + 3 * (CW + GAP) + 16
q4.append(f'<text x="{lx}" y="{58+10}" class="sublab">hurts</text>')
for k in range(11):
    t = (k / 10) * 2 - 1
    q4.append(
        f'<rect x="{lx}" y="{58+18+k*13}" width="15" height="13" fill="{diverging(t*vmax, vmax)}"/>'
    )
q4.append(f'<text x="{lx}" y="{58+18+11*13+14}" class="sublab">helps</text>')
figs.append(
    {
        "n": 4,
        "title": "Elaboration is a leveller: it hurts the leaders and rescues the stragglers",
        "sub": "Per-model change in NDCG@10 on TREC-COVID, averaged over the four rewriters. Rows are "
        "ordered by the model's score on the benchmark's own questions, strongest at the top.",
        "svg": f'<svg viewBox="0 0 {W4} {H4}" class="chart">{"".join(q4)}</svg>',
        "cap": "The verbose column is a gradient: the models that win on the benchmark's short questions "
        "lose ground, and the ones near the bottom gain it &#8212; Spearman between baseline strength "
        "and verbose &#916; is <strong>&#8722;0.887 (p &lt; 0.0001)</strong>. "
        "<code>qwen3-embedding-8b</code> gains +0.265 and moves from 15th to <strong>1st</strong>. "
        "That is the reshuffle mechanism made visible, and it is the strongest argument in the piece "
        "that a leaderboard ranks models for <em>a query style</em>. On CUREv1 the same correlation is "
        "&#8722;0.278 (p = 0.22), so this is a TREC-COVID result, not a universal law.",
        "place": "Strong hero candidate, or open &#167;1 with it.",
    }
)

# ---------------------------------------------------------------- FIG 5
UP, DOWN = "#2a78d6", "#e34948"  # validated diverging poles: gained rank / lost rank
left = sorted(rows, key=lambda r: -r["baseline"])
vtot = {r["model"]: r["baseline"] + (r["verbose"] or 0) for r in rows}
right = sorted(rows, key=lambda r: -vtot[r["model"]])
rL = {r["model"]: i for i, r in enumerate(left)}
rR = {r["model"]: i for i, r in enumerate(right)}

RW, ROW = 940, 25
XL, XR = 292, 596
RH = 74 + len(rows) * ROW
q5: list[str] = []
q5.append(f'<text x="{XL}" y="40" text-anchor="middle" class="axlab">human questions</text>')
q5.append(f'<text x="{XL}" y="56" text-anchor="middle" class="sublab">the benchmark\'s own board</text>')
q5.append(f'<text x="{XR}" y="40" text-anchor="middle" class="axlab">verbose rewrite</text>')
q5.append(f'<text x="{XR}" y="56" text-anchor="middle" class="sublab">mean of 4 rewriters</text>')


def ry(i: int) -> float:
    return 74 + i * ROW + ROW / 2


for r in rows:
    m = r["model"]
    a, b = rL[m], rR[m]
    d = a - b  # positive = moved up the board
    col = UP if d >= 3 else DOWN if d <= -3 else "var(--faint)"
    wgt = 2.6 if abs(d) >= 5 else 2 if abs(d) >= 3 else 1.3
    tip = (
        f"{esc(short(m))} &#183; #{a+1} &#8594; #{b+1} ({d:+d}) &#183; "
        f"NDCG@10 {r['baseline']:.3f} &#8594; {vtot[m]:.3f} ({r['verbose']:+.3f})"
    )
    q5.append(
        f'<path class="ln" d="M{XL+8},{ry(a):.1f} C{XL+110},{ry(a):.1f} {XR-110},{ry(b):.1f} {XR-8},{ry(b):.1f}" '
        f'fill="none" stroke="{col}" stroke-width="{wgt}" opacity="{1 if abs(d)>=3 else .5}" data-tip="{tip}"/>'
    )
    for xx, idx in ((XL + 8, a), (XR - 8, b)):
        q5.append(
            f'<circle class="dot" cx="{xx}" cy="{ry(idx):.1f}" r="4" fill="{col}" '
            f'stroke="var(--surface-1)" stroke-width="2" data-tip="{tip}"/>'
        )
    q5.append(
        f'<text x="{XL-14}" y="{ry(a)+4:.1f}" text-anchor="end" class="rowlab" '
        f'fill="{col if abs(d)>=3 else "var(--text-secondary)"}">{esc(short(m))}</text>'
    )
    q5.append(f'<text x="{XL-6}" y="{ry(a)+4:.1f}" text-anchor="end" class="tick">{a+1}</text>')
    q5.append(f'<text x="{XR+6}" y="{ry(b)+4:.1f}" class="tick">{b+1}</text>')
    lbl = f"{short(m)}  {d:+d}" if abs(d) >= 3 else short(m)
    q5.append(
        f'<text x="{XR+22}" y="{ry(b)+4:.1f}" class="rowlab" '
        f'fill="{col if abs(d)>=3 else "var(--text-secondary)"}">{esc(lbl)}</text>'
    )
figs.append(
    {
        "n": 5,
        "title": "The board, reordered by elaboration alone",
        "sub": "The same 21 models, ranked on TREC-COVID by the benchmark's own questions (left) and by "
        "the verbose rewrite of those same questions (right). Corpus and relevance labels are "
        "identical on both sides &#8212; only the wording of the query changed. "
        "Blue gained at least three places, red lost at least three.",
        "svg": f'<svg viewBox="0 0 {RW} {RH}" class="chart">{"".join(q5)}</svg>',
        "cap": "<code>qwen3-embedding-8b</code> goes from <strong>15th to 1st</strong>; "
        "<code>voyage-law-2</code> climbs five places; four of the top six on the benchmark's own "
        "board slide down. Kendall &tau; for this pair is 0.52&#8211;0.70 depending on the rewriter. "
        "If a reader takes one picture away from the piece, my vote is this one &#8212; it is the "
        "claim in the title, drawn.",
        "place": "Hero. Figure 4 then works as the follow-up that explains <em>why</em> it reorders.",
    }
)

# ---------------------------------------------------------------- page
body = "\n".join(
    f"""<figure class="fig">
  <div class="fignum">Figure {f['n']}</div>
  <h2>{f['title']}</h2>
  <p class="sub">{f['sub']}</p>
  <div class="viz-root plot">{f['svg']}</div>
  <figcaption>{f['cap']}</figcaption>
  <p class="place"><strong>Placement</strong> &#183; {f['place']}</p>
</figure>"""
    for f in figs
)

OUT.write_text(
    f"""<!doctype html>
<html><head><meta charset="utf8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Candidate graphics &#8212; query-form sensitivity post</title>
<style>
:root {{
  --page:#f9f9f7; --surface-1:#fcfcfb; --surface-2:#f0efec; --mid:#f0efec;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --faint:#c9c8c2; --border:rgba(11,11,11,.10);
}}
@media (prefers-color-scheme:dark){{ :root:where(:not([data-theme="light"])){{
  --page:#0d0d0d; --surface-1:#1a1a19; --surface-2:#232322; --mid:#383835;
  --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --faint:#4a4a47; --border:rgba(255,255,255,.10);
}}}}
:root[data-theme="dark"]{{
  --page:#0d0d0d; --surface-1:#1a1a19; --surface-2:#232322; --mid:#383835;
  --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --faint:#4a4a47; --border:rgba(255,255,255,.10);
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--page);color:var(--text-primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;line-height:1.6}}
.wrap{{max-width:900px;margin:0 auto;padding:44px 22px 80px}}
h1{{font-size:26px;margin:0 0 8px;letter-spacing:-.01em}}
.lede{{color:var(--text-secondary);margin:0 0 8px;max-width:70ch}}
.fig{{margin:34px 0 0;padding:22px 22px 18px;background:var(--surface-1);
  border:1px solid var(--border);border-radius:12px}}
.fignum{{font-size:11px;font-weight:640;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}}
h2{{font-size:18px;margin:6px 0 6px;letter-spacing:-.01em}}
.sub{{color:var(--text-secondary);font-size:14px;margin:0 0 14px;max-width:76ch}}
.plot{{margin:6px 0 12px}}
svg.chart{{width:100%;height:auto;overflow:visible;display:block}}
figcaption{{font-size:13.5px;color:var(--text-secondary);border-top:1px solid var(--border);padding-top:12px;max-width:78ch}}
.place{{font-size:12.5px;color:var(--muted);margin:10px 0 0}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:.92em;background:var(--surface-2);padding:1px 5px;border-radius:4px}}
.tick{{font-size:11px;fill:var(--muted)}}
.axlab{{font-size:12.5px;fill:var(--text-secondary);font-weight:560}}
.sublab{{font-size:11px;fill:var(--muted)}}
.rowlab{{font-size:11.5px;fill:var(--text-secondary)}}
.cellval{{font-size:10.5px;font-weight:640}}
.panel{{font-size:12.5px;fill:var(--text-secondary);font-weight:640}}
.dlab{{font-size:12px;font-weight:620}}
.note{{font-size:11px;fill:var(--muted)}}
.ln.bg{{opacity:.42}} .dot.bg{{opacity:.42}}
.ln,.dot,.cell{{transition:opacity .12s}}
.ln:hover,.dot:hover,.cell:hover{{opacity:1;cursor:crosshair}}
#tip{{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--text-primary);
  color:var(--page);font-size:12px;padding:6px 9px;border-radius:6px;max-width:330px;z-index:9}}
</style></head><body>
<div class="wrap">
<h1>Candidate graphics for the query-form post</h1>
<p class="lede">Four options, each built from the run outputs in <code>results/</code>. Hover any mark for its
numbers. Everything renders in light and dark.</p>
{body}
</div>
<div id="tip"></div>
<script>
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{{
  el.addEventListener('mousemove',e=>{{
    tip.innerHTML=el.dataset.tip;tip.style.opacity=1;
    const r=tip.getBoundingClientRect();
    tip.style.left=Math.min(e.clientX+14,innerWidth-r.width-10)+'px';
    tip.style.top=Math.max(e.clientY-r.height-12,8)+'px';
  }});
  el.addEventListener('mouseleave',()=>tip.style.opacity=0);
}});
</script>
</body></html>""",
    encoding="utf8",
)
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
print(f"fig1 models={len(hf)}  fig2 cells={len(grid)}  fig4 rows={len(rows)} vmax={vmax:.3f}")
