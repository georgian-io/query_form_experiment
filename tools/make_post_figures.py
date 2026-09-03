"""Emit the two post figures as standalone, compact, self-contained SVG files.

Each file carries its own <style> (including a dark-mode media query) so it renders
correctly inline, as <img src=...>, or exported to PNG. Shapes are emitted via <use>
and grouped under shared <g> attributes to keep the markup small enough to inline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATA = json.loads(Path(sys.argv[1]).read_text())
OUTDIR = Path(sys.argv[2])
OUTDIR.mkdir(parents=True, exist_ok=True)

RED = ["#a52a2a", "#c8413f", "#de6b66", "#eea19c", "#f8d5d1"]
BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#17518f"]
UP, DOWN, FLAT = "#2a78d6", "#e34948", "#b9b7b0"

STYLE = (
    ".qk{font-size:11px;fill:#898781}"
    ".qs{font-size:11.5px;fill:#898781}"
    ".qc{font-size:13px;fill:#898781;font-weight:640}"
    ".qr{font-size:12px;fill:#6d6b66}"
    ".qv{font-size:10.5px;font-weight:640}"
    ".qh{font-size:14px;fill:#5d5b57;font-weight:650}"
    "@media(prefers-color-scheme:dark){.qr{fill:#a8a69e}.qh{fill:#c3c2b7}}"
)
SVGATTR = (
    ' xmlns="http://www.w3.org/2000/svg" font-family="system-ui,-apple-system,sans-serif"'
    ' style="width:100%;height:auto;display:block"'
)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def short(m: str) -> str:
    return m.replace("sentence-transformers/", "").replace("BAAI/", "").replace("qwen/", "")


def lerp(a: str, b: str, t: float) -> str:
    ar, ag, ab = (int(a[i : i + 2], 16) for i in (1, 3, 5))
    br, bg, bb = (int(b[i : i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (
        round(ar + (br - ar) * t),
        round(ag + (bg - ag) * t),
        round(ab + (bb - ab) * t),
    )


def diverging(v: float, vmax: float) -> str:
    t = max(-1.0, min(1.0, v / vmax))
    if abs(t) < 0.045:
        return "#e8e7e3"
    ramp = (BLUE if t > 0 else RED)[::-1]
    pos = abs(t) * (len(ramp) - 1)
    i = min(int(pos), len(ramp) - 2)
    return lerp(ramp[i], ramp[i + 1], pos - i)


def ink(f: str) -> str:
    r, g, b = (int(f[i : i + 2], 16) for i in (1, 3, 5))
    return "#fff" if (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.55 else "#0b0b0b"


rows = DATA["deltas"]["trec_covid"]
CONDS = ["paraphrase", "terse", "verbose"]
vmax = max(abs(r[c]) for r in rows for c in CONDS)

# ------------------------------------------------------------------ heatmap
by_base = sorted(rows, key=lambda r: -r["baseline"])
LBL, CW, CH, GAP, TOP = 248, 128, 22, 4, 62
W = LBL + 3 * (CW + GAP) + 104
H = TOP + len(by_base) * (CH + GAP) + 30
o = [
    f'<svg{SVGATTR} viewBox="0 0 {W} {H}"><style>{STYLE}</style>',
    f'<defs><rect id="qfc" width="{CW}" height="{CH}" rx="3"/><rect id="qfg" width="14" height="12"/></defs>',
    '<text class="qh" y="20">Change in NDCG@10 when the query is rewritten</text>',
    '<text class="qs" y="38">TREC-COVID &#183; 21 models &#183; mean of four LLM rewriters &#183; corpus and relevance labels held fixed</text>',
]
o.append('<g class="qc" text-anchor="middle">')
for j, c in enumerate(CONDS):
    o.append(f'<text x="{LBL + j*(CW+GAP) + CW//2}" y="{TOP-8}">{c}</text>')
o.append("</g>")
o.append(f'<text class="qs" x="{LBL-10}" y="{TOP-8}" text-anchor="end">strongest first &#8595;</text>')
o.append('<g class="qr" text-anchor="end">')
for i, r in enumerate(by_base):
    o.append(f'<text x="{LBL-10}" y="{TOP + i*(CH+GAP) + CH//2 + 4}">{esc(short(r["model"]))}</text>')
o.append("</g>")
for i, r in enumerate(by_base):
    y = TOP + i * (CH + GAP)
    for j, c in enumerate(CONDS):
        o.append(f'<use href="#qfc" x="{LBL + j*(CW+GAP)}" y="{y}" fill="{diverging(r[c], vmax)}"/>')
o.append('<g class="qv" text-anchor="middle">')
for i, r in enumerate(by_base):
    for j, c in enumerate(CONDS):
        if abs(r[c]) >= 0.085:
            f = diverging(r[c], vmax)
            o.append(
                f'<text x="{LBL + j*(CW+GAP) + CW//2}" y="{TOP + i*(CH+GAP) + CH//2 + 4}" fill="{ink(f)}">{r[c]:+.2f}</text>'
            )
o.append("</g>")
lx = LBL + 3 * (CW + GAP) + 18
o.append(f'<text class="qs" x="{lx}" y="{TOP+9}">hurts</text>')
for k in range(11):
    o.append(f'<use href="#qfg" x="{lx}" y="{TOP+16+k*12}" fill="{diverging(((k/10)*2-1)*vmax, vmax)}"/>')
o.append(f'<text class="qs" x="{lx}" y="{TOP+16+11*12+12}">helps</text></svg>')
(OUTDIR / "fig-leveller-heatmap.svg").write_text("".join(o), encoding="utf8")

# --------------------------------------------------------------- bump chart
vtot = {r["model"]: r["baseline"] + r["verbose"] for r in rows}
right = sorted(rows, key=lambda r: -vtot[r["model"]])
rL = {r["model"]: i for i, r in enumerate(by_base)}
rR = {r["model"]: i for i, r in enumerate(right)}
ROW, XL, XR, TOP2 = 25, 286, 592, 78
W2, H2 = 930, TOP2 + len(rows) * ROW + 24


def yy(i: int) -> int:
    return TOP2 + i * ROW + ROW // 2


b = [
    f'<svg{SVGATTR} viewBox="0 0 {W2} {H2}"><style>{STYLE}</style>',
    '<defs><circle id="qfd" r="4"/></defs>',
    '<text class="qh" y="20">The same 21 models, ranked twice</text>',
    '<text class="qs" y="38">TREC-COVID &#183; only the wording of the query differs between the two columns</text>',
    f'<g class="qc" text-anchor="middle"><text x="{XL}" y="{TOP2-24}">human questions</text>'
    f'<text x="{XR}" y="{TOP2-24}">verbose rewrite</text></g>',
    f'<g class="qs" text-anchor="middle"><text x="{XL}" y="{TOP2-9}">the benchmark\'s own board</text>'
    f'<text x="{XR}" y="{TOP2-9}">mean of four rewriters</text></g>',
]
groups: dict[tuple[str, float, str], list[str]] = {}
for r in rows:
    m = r["model"]
    a, z = rL[m], rR[m]
    d = a - z
    col = UP if d >= 3 else DOWN if d <= -3 else FLAT
    key = (col, 2.4 if abs(d) >= 3 else 1.3, "1" if abs(d) >= 3 else "0.55")
    groups.setdefault(key, []).append(
        f'<path d="M{XL+8},{yy(a)} C{XL+112},{yy(a)} {XR-112},{yy(z)} {XR-8},{yy(z)}"/>'
    )
for (col, w, op), items in groups.items():
    b.append(f'<g fill="none" stroke="{col}" stroke-width="{w}" opacity="{op}">{"".join(items)}</g>')
dots: dict[tuple[str, str], list[str]] = {}
for r in rows:
    m = r["model"]
    a, z = rL[m], rR[m]
    d = a - z
    col = UP if d >= 3 else DOWN if d <= -3 else FLAT
    key = (col, "1" if abs(d) >= 3 else "0.55")
    dots.setdefault(key, []).append(
        f'<use href="#qfd" x="{XL+8}" y="{yy(a)}"/><use href="#qfd" x="{XR-8}" y="{yy(z)}"/>'
    )
for (col, op), items in dots.items():
    b.append(f'<g fill="{col}" opacity="{op}">{"".join(items)}</g>')
b.append('<g class="qk" text-anchor="end">')
for r in rows:
    b.append(f'<text x="{XL-8}" y="{yy(rL[r["model"]])+4}">{rL[r["model"]]+1}</text>')
b.append('</g><g class="qk">')
for r in rows:
    b.append(f'<text x="{XR+8}" y="{yy(rR[r["model"]])+4}">{rR[r["model"]]+1}</text>')
b.append("</g>")
lt, rt = [], []
for r in rows:
    m = r["model"]
    a, z = rL[m], rR[m]
    d = a - z
    col = UP if d >= 3 else DOWN if d <= -3 else None
    f = f' fill="{col}"' if col else ""
    lt.append(f'<text x="{XL-30}" y="{yy(a)+4}"{f}>{esc(short(m))}</text>')
    lab = f"{short(m)}   {d:+d}" if col else short(m)
    rt.append(f'<text x="{XR+30}" y="{yy(z)+4}"{f}>{esc(lab)}</text>')
b.append(f'<g class="qr" text-anchor="end">{"".join(lt)}</g>')
b.append(f'<g class="qr">{"".join(rt)}</g></svg>')
(OUTDIR / "fig-board-reorder.svg").write_text("".join(b), encoding="utf8")

for f in sorted(OUTDIR.glob("fig-*.svg")):
    print(f"{f}  {f.stat().st_size/1024:.1f} KB")

# ----------------------------------------------------- human phrasings bump
hf = DATA["human_fields"]
FLDS = [("keyword", "keyword"), ("question", "question"), ("narrative", "narrative")]
rank = {
    k: {r["model"]: i for i, r in enumerate(sorted(hf, key=lambda r: -r[k]))} for k, _ in FLDS
}
ROW3, TOP3 = 25, 92
X3 = [300, 512, 724]
W3, H3 = 946, TOP3 + len(hf) * ROW3 + 24


def y3(i: int) -> int:
    return TOP3 + i * ROW3 + ROW3 // 2


h = [
    f'<svg{SVGATTR} viewBox="0 0 {W3} {H3}"><style>{STYLE}</style>',
    '<defs><circle id="qfd" r="4"/></defs>',
    '<text class="qh" y="20">Three human phrasings of the same need, ranked</text>',
    '<text class="qs" y="38">TREC-COVID &#183; 20 models &#183; every phrasing written by people, judged against the same labels</text>',
    '<text class="qs" y="56">Blue climbs as the query gets longer; red climbs as it gets shorter.</text>',
]
h.append('<g class="qc" text-anchor="middle">')
for i, (_, lab) in enumerate(FLDS):
    h.append(f'<text x="{X3[i]}" y="{TOP3-30}">{lab}</text>')
h.append("</g>")
h.append(
    f'<g class="qs" text-anchor="middle"><text x="{X3[0]}" y="{TOP3-15}">2 words</text>'
    f'<text x="{X3[1]}" y="{TOP3-15}">one sentence</text>'
    f'<text x="{X3[2]}" y="{TOP3-15}">~22 words</text></g>'
)
segs: dict[tuple[str, float, str], list[str]] = {}
dots3: dict[tuple[str, str], list[str]] = {}
for r in hf:
    m = r["model"]
    a, b, c = rank["keyword"][m], rank["question"][m], rank["narrative"][m]
    d = a - c
    col = UP if d >= 3 else DOWN if d <= -3 else FLAT
    key = (col, 2.4 if abs(d) >= 3 else 1.3, "1" if abs(d) >= 3 else "0.5")
    for (p, q), (xa, xb) in zip(((a, b), (b, c)), ((X3[0], X3[1]), (X3[1], X3[2]))):
        segs.setdefault(key, []).append(
            f'<path d="M{xa+8},{y3(p)} C{xa+78},{y3(p)} {xb-78},{y3(q)} {xb-8},{y3(q)}"/>'
        )
    dk = (col, "1" if abs(d) >= 3 else "0.5")
    for x, idx in zip(X3, (a, b, c)):
        dots3.setdefault(dk, []).append(f'<use href="#qfd" x="{x}" y="{y3(idx)}"/>')
for (col, w, op), items in segs.items():
    h.append(f'<g fill="none" stroke="{col}" stroke-width="{w}" opacity="{op}">{"".join(items)}</g>')
for (col, op), items in dots3.items():
    h.append(f'<g fill="{col}" opacity="{op}">{"".join(items)}</g>')
h.append('<g class="qk" text-anchor="end">')
for r in hf:
    h.append(f'<text x="{X3[0]-8}" y="{y3(rank["keyword"][r["model"]])+4}">{rank["keyword"][r["model"]]+1}</text>')
h.append('</g><g class="qk">')
for r in hf:
    h.append(f'<text x="{X3[2]+8}" y="{y3(rank["narrative"][r["model"]])+4}">{rank["narrative"][r["model"]]+1}</text>')
h.append("</g>")
lt3, rt3 = [], []
for r in hf:
    m = r["model"]
    a, c = rank["keyword"][m], rank["narrative"][m]
    d = a - c
    col = UP if d >= 3 else DOWN if d <= -3 else None
    f = f' fill="{col}"' if col else ""
    lt3.append(f'<text x="{X3[0]-30}" y="{y3(a)+4}"{f}>{esc(short(m))}</text>')
    lab = f"{short(m)}   {d:+d}" if col else short(m)
    rt3.append(f'<text x="{X3[2]+30}" y="{y3(c)+4}"{f}>{esc(lab)}</text>')
h.append(f'<g class="qr" text-anchor="end">{"".join(lt3)}</g>')
h.append(f'<g class="qr">{"".join(rt3)}</g></svg>')
(OUTDIR / "fig-human-phrasings.svg").write_text("".join(h), encoding="utf8")
print(f'{OUTDIR / "fig-human-phrasings.svg"}  {(OUTDIR / "fig-human-phrasings.svg").stat().st_size/1024:.1f} KB')
