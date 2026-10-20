"""Write fig-heldout.pdf (blind detection on GPUBugBench): timing cases per blind round with the strict paired and
root-cause counts, and the pooled rates with Wilson 95% intervals; structure and meaning in grey, as categories outside
the synchronization guarantee. Counts: heldout.json (the stored per-round results of rounds R1-R6).
All text prints at 6 pt or larger: the PDF is exactly \\columnwidth wide, so LaTeX does not rescale it.
Run: python3 build_heldout.py [outdir]   (needs matplotlib and the Libertinus Sans font)"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
D = json.loads((HERE / "heldout.json").read_text())
R, T = D["rounds"], D["total"]
assert (T["timing"]["detected"], T["timing"]["cases"]) == (21, 35)
assert sum(r["root_cause"] for r in R) == 24 and sum(r["timing_cases"] for r in R) == 35

INK, MUTED, GRID, GREY = "#2b2f36", "#6b7480", "#dfe7ee", "#9aa3ad"
TIMING, TIMING_L = "#2c6aa8", "#cfdff0"
plt.rcParams.update({
    "font.family": "Libertinus Sans", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": "#9fb2c0", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})
COL = 3.33   # ACM single column, inches


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL, 1.9), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.1})
for i, r in enumerate(R):
    n, rc, k = r["timing_cases"], r["root_cause"], r["strict"]
    ax.bar(i, n, 0.66, color="white", edgecolor=TIMING, linewidth=0.5)
    ax.bar(i, rc, 0.66, color=TIMING_L, edgecolor=TIMING, linewidth=0.4)
    ax.bar(i, k, 0.66, color=TIMING, linewidth=0)
    ax.text(i, n + 0.2, f"{k}/{n}", ha="center", va="bottom", fontsize=6.6, color=INK)
ax.set_ylim(0, 9.6); ax.set_yticks([0, 2, 4, 6, 8]); ax.set_ylabel("timing cases", labelpad=2)
ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
ax.set_xticks(range(len(R)), [f"R{r['round']}" for r in R])
ax.set_xlabel("blind round (checker frozen per round)", labelpad=1.5)
ax.legend(handles=[Patch(facecolor=TIMING, label="strict paired"),
                   Patch(facecolor=TIMING_L, edgecolor=TIMING, linewidth=.4, label="root cause"),
                   Patch(facecolor="white", edgecolor=TIMING, linewidth=.5, label="cases")],
          ncol=3, loc="lower left", bbox_to_anchor=(-0.2, 1.0), handlelength=1.0, handletextpad=0.4,
          columnspacing=0.8, fontsize=6.8, borderaxespad=0.2)

root = sum(r["root_cause"] for r in R)
rows = [("timing, strict paired", (T["timing"]["detected"], 35), 3.3, TIMING, True),
        ("timing, root cause", (root, 35), 2.25, TIMING, True),
        ("structure, strict", (T["structure"]["detected"], T["structure"]["cases"]), 0.95, GREY, False),
        ("meaning, strict", (T["meaning"]["detected"], T["meaning"]["cases"]), 0.0, GREY, False)]
for name, (k, n), y, col, main in rows:
    lo, hi = wilson(k, n)
    bx.plot([lo * 100, hi * 100], [y, y], color=col, lw=1.6 if main else 1.1, solid_capstyle="butt")
    bx.plot([100 * k / n], [y], "o", ms=4.2 if main else 3.2, color=col, mec="white", mew=0.6)
    bx.text(1, y + 0.42, name, ha="left", va="center", fontsize=6.8 if main else 6.6,
            color=INK if main else MUTED, fontweight="bold" if main else "normal")
    bx.text(hi * 100 + 3, y, f"{k}/{n}", ha="left", va="center", fontsize=6.8 if main else 6.6,
            color=INK if main else MUTED)
bx.text(1, 1.76, "outside the sync. guarantee:", ha="left", va="center", fontsize=6.4, color=MUTED, style="italic")
bx.set_xlim(0, 100); bx.set_xticks([0, 50, 100]); bx.set_ylim(-0.45, 3.95); bx.set_yticks([])
bx.xaxis.grid(True, color=GRID, linewidth=0.5); bx.set_axisbelow(True)
bx.spines["left"].set_visible(False)
bx.set_xlabel("blind, as run (%), Wilson 95%", labelpad=1.5)

# make the tight bounding box exactly COL wide, so the figure prints at its nominal size
for _ in range(6):
    bw = fig.get_tightbbox(fig.canvas.get_renderer()).width + 0.04
    if abs(bw - COL) < 0.003:
        break
    w, h = fig.get_size_inches()
    fig.set_size_inches(w + (COL - bw), h)
fig.savefig(OUT / "fig-heldout.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig-heldout.png", bbox_inches="tight", pad_inches=0.02, dpi=400)
print("wrote", OUT / "fig-heldout.pdf")
