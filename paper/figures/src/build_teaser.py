"""Fill fig1-teaser.template.html with verbatim checker output and write fig1-teaser.html (Figure 1).

Inputs (all in ../teaser-evidence, produced by the checker frozen for the fresh studies, hash 7cb216ed5753):
  bug-check.txt   check_trace.py on SDL3 testsprite (commit e4af6f8e), 20 presents, conservative capture
  bug-trace.jsonl.gz  the recorded trace (sequence numbers in panel 2)
  fix-check.txt   the same after experiments/repairs/patches/SDL-1.diff
  bug-validation-log.txt  lavapipe + VVL 0ad67fd (HEAD-max syncval) on the unpatched program, 20 presents
Run: python3 build_teaser.py && NODE_PATH=/opt/node-tools/node_modules node render.js fig1-teaser
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = HERE.parent / "teaser-evidence"

bug = (EV / "bug-check.txt").read_text().splitlines()
summary = json.loads(bug[-1])
assert summary["covenant_version"] == "7cb216ed5753" and summary["warnings"] == 0 and summary["presents"] == 20
i = next(k for k, l in enumerate(bug) if l.startswith("[timing] timing.cross-queue-race"))
diag = bug[i:i + 4]
assert diag[1].strip().startswith("at vkCmdDraw @seq 1139") and "before call seq 1442" in diag[2]
n = sum(l.startswith("[timing] ") for l in bug)
assert n == sum(l.startswith("[timing] timing.cross-queue-race") for l in bug)
fixed = (EV / "fix-check.txt").read_text().splitlines()
assert fixed[0] == "all checks passed" and json.loads(fixed[-1])["covenant_version"] == "7cb216ed5753"
sv = (EV / "bug-validation-log.txt").read_text().strip()
assert sv == "" and "presents=20" in (EV / "bug-validation-stdout.txt").read_text()


def esc(lines, hl=()):
    s = html.escape("\n".join(lines), quote=False)
    for h in hl:
        s = s.replace(h, f'<span class="rf0">{h}</span>')
    return s


t = (HERE / "fig1-teaser.template.html").read_text()
t = t.replace("{{DIAG}}", esc(diag, ["'buffer#1'", "vkCmdDraw @seq 1139", "host write to buffer#1"]))
t = t.replace("{{NBUG}}", str(n)).replace("{{FIXED}}", "all checks passed").replace("{{SVBUG}}", "no message")
APPS = [  # (image, object-position, name, finding, status class, status) -- facts from Sections 7.4-7.6, Figures 8-9
    ("sdl_testsprite", "50% 50%", "SDL3 &middot; sprite test", "this race", "ok2", "patched: 166 &rarr; 0"),
    ("vkmark_vertex", "50% 15%", "vkmark &middot; 7 scenes", "5 causes", "ok2", "patched: 1,124 &rarr; 0"),
    ("q3d_reflectionprobes_c", "50% 45%", "Qt Quick 3D &middot; 14 programs", "study 1", "up", "3 true causes"),
    ("gtk_widget_factory", "50% 20%", "GTK 4 &middot; 9 programs", "study 2", "up", "3 true causes"),
    ("mpv_gpunext", "50% 50%", "mpv / libplacebo &middot; 4 configs", "study 2", "up", "1 true cause"),
]
apps_html = "\n".join(
    f'<div class="app"><div class="im"><img src="img/{i}.jpg" style="object-position:{op}"><div class="nm">{n}</div></div>'
    f'<div class="st"><span class="dm">{f} &middot; </span><span class="{c}">{st}</span></div></div>' for i, op, n, f, c, st in APPS)
t = t.replace("{{APPS}}", apps_html)
import re; assert not re.search(r"\{\{[A-Z0-9]+\}\}", t)
(HERE / "fig1-teaser.html").write_text(t)
(HERE / "fig1-teaser.diagnostics.txt").write_text("\n".join(diag) + f"\n\noccurrences: {n}\nfixed: all checks passed\n"
                                                  "validation: no message\n")
print("wrote fig1-teaser.html; occurrences", n)
