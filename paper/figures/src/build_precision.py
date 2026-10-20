"""Write fig-precision.html (Figure 7): the two fresh studies, one row per code base. Each row shows frames that the
code base's programs render (lavapipe captures of the same commands, cap.py), the share of timing report groups per
class for the checker frozen before the study (conservative and observed capture) and for the release on the same
traces (post-hoc), and how many true root causes it has with their upstream status
(experiments/fresh-studies/UPSTREAM.md; the causes are named in the paper's tables and Supplement I).
Counts: precision.json (prep_data.py, from experiments/fresh-studies/data/classes-*.json; gtk_path_walk excluded:
not grouped when frozen, grouping incomplete in the release).
Run: python3 build_precision.py && FIGOUT=.. NODE_PATH=/opt/node-tools/node_modules node render.js fig-precision"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'precision.json').read_text())
CLS = [('TRUE', 'true', '#2f6fa8'), ('TECH', 'technicality', '#9cc3e4'), ('UNRESOLVED', 'unresolved', '#7f8892'),
       ('FA', 'false alarm', 'url(#fa)')]

BASES = [  # study, key, name, revision, images, causes changed upstream before the study, still present, note
    (1, 'qt', 'Qt Quick 3D', 'Qt 6.4.2', ['q3d_reflectionprobes', 'q3d_morphing'], 2, 1,
     'false alarms: host writes into per-frame slots smaller than a page (conservative capture)'),
    (1, 'wgpu', 'wgpu', None, ['wgpu_water', 'wgpu_shadow'], 0, 0,
     'false alarms: three checker defects, fixed after the study'),
    (1, 'imgui', 'Dear ImGui', '9d9a621f', ['imgui_glfw', 'imgui_sdl2'], 0, 1, None),
    (1, 'vkcube', 'vkcube', '1.3.275', ['vkcube_w'], 1, 0, None),
    (2, 'gtk', 'GTK 4', '4.14.5', ['gtk_iconscroll', 'gtk_widget_factory'], 3, 0,
     'false alarms: a frame\'s images in one descriptor array (frozen); atlas rectangles (release)'),
    (2, 'vkmark', 'vkmark', 'ab6e6f3', ['vkmark_vertex', 'vkmark_desktop'], 2, 3, None),
    (2, 'mpv', 'mpv / libplacebo', '0.37 / 6.338', ['mpv_gpunext', 'mpv_gpu_hq'], 1, 0, None),
    (2, 'gst', 'GStreamer', '1.24.2', ['gst_sink_w'], 0, 2, None),
]
UNIT = {'qt': 'programs', 'wgpu': 'examples', 'imgui': 'programs', 'vkcube': 'program', 'gtk': 'programs',
        'vkmark': 'scenes', 'mpv': 'configurations', 'gst': 'pipelines'}
UP, OPEN = '#3f6c86', '#b4532a'


def bar(c, label, post=False):
    tot = sum(c.values())
    if tot == 0:
        return (f'<div class="br{" post" if post else ""}"><span class="bl">{label}</span><div class="bb empty">no report</div>'
                f'<span class="bn"><b>&ndash;</b> 0 groups</span></div>')
    x, rects = 0.0, ''
    for k, _, col in CLS:
        w = 100 * c.get(k, 0) / tot
        if w:
            rects += f'<rect x="{x:.3f}" y="0" width="{w:.3f}" height="10" fill="{col}"/>'
            x += w
    dash = ' stroke-dasharray="2.2 1.4"' if post else ''
    svg = (f'<svg class="bb" viewBox="0 0 100 10" preserveAspectRatio="none">{rects}'
           f'<rect x="0.25" y="0.25" width="99.5" height="9.5" fill="none" stroke="#56636f" stroke-width="0.5"'
           f' vector-effect="non-scaling-stroke"{dash}/></svg>')
    nums = ' / '.join(f'{c.get(k, 0):,}' for k in ('TRUE', 'TECH', 'FA'))
    if c.get('UNRESOLVED'):
        nums += f' + {c["UNRESOLVED"]} unres.'
    pct = 100 * c.get('TRUE', 0) / tot
    return (f'<div class="br{" post" if post else ""}"><span class="bl">{label}</span>{svg}'
            f'<span class="bn"><b>{pct:.0f}%</b> {nums}</span></div>')


def row(b):
    st, key, name, rev, imgs, up, still, note = b
    n = P['progs'][key]
    im = ''.join(f'<img src="img/{i}.jpg">' for i in imgs)
    fr, fo, rc = (P.get(f'{a}/{m}/{key}', {}) for a, m in (('frozen', 'conservative'), ('frozen', 'observed'),
                                                             ('release', 'conservative')))
    k = up + still
    parts = []
    if up:
        parts.append(f'<span style="color:{UP}">&uarr;&#8239;{up}</span>')
    if still:
        parts.append(f'<span style="color:{OPEN}">&#9679;&#8239;{still}</span>')
    causes = (f'<b>{k}</b> true cause{"s" if k != 1 else ""}' + (f' &nbsp;{" ".join(parts)}' if parts else ''))
    nt = f'<div class="nt">{note}</div>' if note else ''
    return (f'<div class="row"><div class="ims n{len(imgs)}">{im}</div>'
            f'<div class="who"><div class="nm">{name}</div><div class="sub">{n}{" + 1" if key == "gtk" else ""} {UNIT[key]}{" &middot; " + rev if rev else ""}</div>'
            f'<div class="cz">{causes}</div></div>'
            f'<div class="bars">{bar(fr, "frozen")}{bar(fo, "observed")}{bar(rc, "release", True)}{nt}</div></div>')


blocks = ''
for s, title in ((1, 'Study 1: 32 programs, checker frozen before its first run'),
                 (2, 'Study 2: 25 programs, a later checker frozen before its first run')):
    blocks += f'<div class="sh">{title}</div>' + ''.join(row(b) for b in BASES if b[0] == s)
sw = ''.join(f'<span class="lg"><svg class="sw" viewBox="0 0 10 8"><rect width="10" height="8" fill="{col}" '
             f'stroke="#56636f" stroke-width="0.6"/></svg>{n}</span>' for _, n, col in CLS)
legend = (f'<div class="lgl">{sw}<span class="lg">bars: share of report groups; numbers: % true, then '
          f'true / technicality / false alarm</span></div>'
          '<div class="lgl"><span class="lg"><b>frozen</b>: the fresh result (conservative capture, the theorem\'s '
          'configuration)</span><span class="lg"><b>observed</b>: same checker, observed capture</span>'
          '<span class="lg"><i>release</i> (dashed): post-hoc, same traces</span>'
          f'<span class="lg"><span style="color:{UP}">&uarr;</span> changed upstream before the study</span>'
          f'<span class="lg"><span style="color:{OPEN}">&#9679;</span> still present upstream</span></div>')
html = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>Fresh studies by code base</title>
<!-- GENERATED by build_precision.py; edit that, not this file. -->
<link rel="stylesheet" href="style.css">
<style>
  #fig {{ width: 672px; font-family: 'Libertinus Sans', sans-serif; }}
  .wrap {{ border: 1px solid var(--panel-border); border-radius: 7px; background: var(--band-bg); padding: 4px 6px 6px; }}
  .sh {{ font-size: 10px; font-weight: 700; margin: 3px 1px 2px; color: #2f4656; }}
  .row {{ display: grid; grid-template-columns: 112px 128px minmax(0, 1fr); gap: 8px; align-items: center;
          background: #fff; border: 1px solid var(--panel-border); border-radius: 4px; padding: 3px 6px 3px 3px;
          margin-bottom: 2px; }}
  .ims {{ display: flex; gap: 2px; }}
  .ims img {{ flex: 1 1 0; width: 0; min-width: 0; aspect-ratio: 4/3; object-fit: cover; border-radius: 2px; display: block; }}
  .ims.n1 img {{ aspect-ratio: 8/3; }}
  .nm {{ font-size: 10.5px; font-weight: 700; line-height: 1.15; }}
  .sub {{ font-size: 8.4px; color: var(--muted); line-height: 1.3; }}
  .cz {{ font-size: 8.6px; margin-top: 2px; }} .cz b {{ font-weight: 700; }}
  .br {{ display: grid; grid-template-columns: 42px minmax(0, 1fr) 162px; gap: 6px; align-items: center;
         font-size: 8.4px; line-height: 11px; margin: 0.4px 0; }}
  .br .bl {{ color: var(--muted); }} .br.post .bl {{ font-style: italic; }}
  .bb {{ width: 100%; height: 9px; display: block; }}
  .bb.empty {{ height: 9px; font-size: 8px; line-height: 8px; color: var(--ok); padding-left: 4px;
               border: 0.6px solid #9cc7aa; border-radius: 1px; }}
  .bn {{ white-space: nowrap; color: var(--ink); }} .bn b {{ font-weight: 700; display: inline-block; min-width: 25px; }}
  .nt {{ font-size: 8.2px; line-height: 10.5px; color: var(--muted); font-style: italic; }}
  .lgl {{ display: flex; flex-wrap: wrap; gap: 2px 12px; font-size: 8.4px; color: var(--ink); margin-top: 3px; }}
  .lg {{ white-space: nowrap; }}
  .sw {{ width: 11px; height: 8px; vertical-align: -0.5px; margin-right: 3px; }}
</style></head><body>
<svg width="0" height="0" style="position:absolute"><defs>
  <pattern id="fa" width="2.4" height="2.4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <rect width="2.4" height="2.4" fill="#ffffff"/><rect width="0.9" height="2.4" fill="#8e98a3"/></pattern>
</defs></svg>
<div id="fig"><div class="wrap">{blocks}{legend}</div>
<div class="caption" id="cap">Fig. (preview)</div></div></body></html>'''
(HERE / 'fig-precision.html').write_text(html)
tot = {k: sum(P.get(f'frozen/conservative/{b[1]}', {}).get(k, 0) for b in BASES if b[0] == 1) for k in ('TRUE', 'TECH', 'FA')}
assert tot == {'TRUE': 130, 'TECH': 12, 'FA': 1971}, tot
assert sum(P['progs'][b[1]] for b in BASES if b[0] == 1) == 32 and sum(P['progs'][b[1]] for b in BASES if b[0] == 2) == 24
print('wrote fig-precision.html')
