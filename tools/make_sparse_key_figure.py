"""Emit the sparse-answer-key illustration as a standalone, self-contained SVG.

Illustrative, not data-driven: the two rankings and the four scores are the worked
A/B example from the post's label-density section. Carries its own <style> with a
dark-mode query, and namespaces every class (.j*) so nothing leaks into the host page
when the file is inlined alongside the other post figures.

    uv run python tools/make_sparse_key_figure.py docs/figures
"""
from __future__ import annotations

import sys
from pathlib import Path

OUTDIR = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

GOLD, RELEVANT = "#17518f", "#5598e7"

STYLE = (
    ".ju{fill:#e5e3de}"
    f".jg{{fill:{GOLD}}}"
    f".jr{{fill:{RELEVANT}}}"
    ".jd{stroke:#c9c7c0;stroke-width:1;stroke-dasharray:3 3}"
    ".jh{font-size:12.5px;fill:#5d5b57;font-weight:650}"
    ".jl{font-size:12px;fill:#6d6b66}"
    ".jn{font-size:11px;fill:#898781}"
    ".jv{font-size:12.5px;fill:#6d6b66;font-weight:640}"
    ".jk{font-size:10px;fill:#98968f}"
    "@media(prefers-color-scheme:dark){"
    ".ju{fill:#3a3934}.jd{stroke:#4d4b46}.jh{fill:#c3c2b7}"
    ".jl{fill:#a8a69e}.jv{fill:#a8a69e}}"
)
SVGATTR = (
    ' xmlns="http://www.w3.org/2000/svg" font-family="system-ui,-apple-system,sans-serif"'
    ' style="width:100%;height:auto;display:block"'
)

SLOT_W, SLOT_H, GAP = 24, 22, 3
ROW_STEP = SLOT_H + 8
X_SLOTS = 58
X_MISS = X_SLOTS + 10 * SLOT_W + 20
X_SCORE = X_MISS + 2 * SLOT_W + 28
WIDTH = X_SCORE + 58

# (label, {rank: class} inside the top-10, n relevant docs missed entirely, score)
PANELS = [
    (
        "Scenario 1 - one document judged relevant",
        [("Model A", {1: "jg"}, 0, "1.000"), ("Model B", {1: "jg"}, 0, "1.000")],
        "Both models put the judged document first. Ranks 2-10 are unlabelled, so the "
        "metric cannot tell these two rankings apart.",
    ),
    (
        "Scenario 2 - the same rankings, three documents judged relevant",
        [("Model A", {1: "jg", 2: "jr", 4: "jr"}, 0, "0.967"),
         ("Model B", {1: "jg"}, 2, "0.469")],
        "A had the other two relevant documents at ranks 2 and 4 all along. B never "
        "retrieved them - they fall outside its top 10 entirely. Both models return "
        "the same ten results in either scenario; all that changed between the two "
        "is how many documents were judged.",
    ),
]


def wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def row(y: int, label: str, marked: dict[int, str], missed: int, score: str) -> str:
    out = [f'<text class="jl" x="0" y="{y + 16}">{label}</text>']
    for rank in range(1, 11):
        x = X_SLOTS + (rank - 1) * SLOT_W
        out.append(f'<rect class="{marked.get(rank, "ju")}" x="{x}" y="{y}" '
                   f'width="{SLOT_W - GAP}" height="{SLOT_H}" rx="2.5"/>')
    for i in range(2):
        x = X_MISS + i * SLOT_W
        cls = "jr" if i < missed else "ju"
        op = "" if i < missed else ' opacity="0.35"'
        out.append(f'<rect class="{cls}" x="{x}" y="{y}"{op} width="{SLOT_W - GAP}" '
                   f'height="{SLOT_H}" rx="2.5"/>')
    out.append(f'<text class="jv" x="{X_SCORE}" y="{y + 16}">{score}</text>')
    return "".join(out)


def build() -> str:
    parts, y = [], 12
    parts.append(f'<text class="jk" x="{X_SLOTS}" y="{y}">the top 10 results, rank 1 to 10</text>')
    parts.append(f'<text class="jk" x="{X_MISS}" y="{y}">not retrieved</text>')
    parts.append(f'<text class="jk" x="{X_SCORE}" y="{y}">NDCG@10</text>')

    for title, rows, note in PANELS:
        y += 24
        parts.append(f'<text class="jh" x="0" y="{y}">{title}</text>')
        y += 10
        top = y
        for label, marked, missed, score in rows:
            parts.append(row(y, label, marked, missed, score))
            y += ROW_STEP
        divider_x = X_MISS - 10
        parts.append(f'<line class="jd" x1="{divider_x}" y1="{top - 4}" '
                     f'x2="{divider_x}" y2="{y - 4}"/>')
        y += 8
        for line in wrap(note, 68):
            parts.append(f'<text class="jn" x="0" y="{y}">{line}</text>')
            y += 14

    y += 12
    lx = 0
    for cls, text in (("jg", "the judged gold"), ("jr", "also relevant"),
                      ("ju", "unjudged")):
        parts.append(f'<rect class="{cls}" x="{lx}" y="{y - 9}" width="11" height="11" rx="2"/>')
        parts.append(f'<text class="jk" x="{lx + 15}" y="{y}">{text}</text>')
        lx += 15 + len(text) * 5.3 + 22
    y += 8

    return (f'<svg viewBox="0 0 {WIDTH} {y}"{SVGATTR}><style>{STYLE}</style>'
            + "".join(parts) + "</svg>")


out = OUTDIR / "fig-sparse-key.svg"
out.write_text(build(), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size} bytes)")
