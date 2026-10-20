"""Write fig-precision.html: the two fresh studies by code base. Each card shows frames of the code base's programs as
they render (lavapipe captures of the same commands, cap.py), the share of timing report groups per class for the
checker frozen before the study (conservative and observed capture) and for the release on the same traces
(post-hoc), and the true root causes with their upstream status (experiments/fresh-studies/UPSTREAM.md).
Counts: precision.json, computed from experiments/fresh-studies/data/classes-*.json (gtk_path_walk excluded: not
grouped when frozen, grouping incomplete in the release).
Run: python3 build_precision.py && FIGOUT=.. NODE_PATH=/opt/node-tools/node_modules node render.js fig-precision"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'precision.json').read_text())
CLS = [('TRUE', 'true', '#2f6fa8'), ('TECH', 'technicality', '#9cc3e4'), ('UNRESOLVED', 'unresolved', '#b9bfc6'),
       ('FA', 'false alarm', '#f4f5f7')]

BASES = [  # study, key, name, programs label, revision, images, causes [(text, status)], note
    (1, 'qt', 'Qt Quick 3D', '14 programs', 'Qt 6.4.2', ['q3d_reflectionprobes', 'q3d_morphing'],
     [('acquire semaphore reused', 'up'), ('present semaphore reused', 'up'),
      ('texture visible to the vertex stage only, sampled by the fragment stage', 'open')],
     'false alarms: host writes into per-frame slots smaller than a page (conservative capture)'),
    (1, 'wgpu', 'wgpu', '15 examples', None, ['wgpu_water', 'wgpu_shadow'], [],
     'false alarms: three checker defects, fixed after the study; the release reports nothing'),
    (1, 'imgui', 'Dear ImGui', '2 programs', '9d9a621f', ['imgui_glfw', 'imgui_sdl2'],
     [('swapchain images transitioned before they are acquired', 'open')], None),
    (1, 'vkcube', 'vkcube', '1 program', '1.3.275', ['vkcube_w'], [('present semaphore reused', 'up')], None),
    (2, 'gtk', 'GTK 4', '8 + 1 programs', '4.14.5', ['gtk_iconscroll', 'gtk_widget_factory'],
     [('rendering never waits for the acquire semaphore', 'up'), ('present waits for acquire, never for rendering', 'up'),
      ('acquire semaphore re-signalled while a present may wait', 'up')],
     'false alarms: all of a frame\'s images in one descriptor array (frozen); atlas rectangles (6, release)'),
    (2, 'vkmark', 'vkmark', '7 scenes', 'ab6e6f3', ['vkmark_vertex', 'vkmark_desktop'],
     [('host rewrites the uniform buffer in flight', 'up'), ('acquire semaphore reused', 'up'),
      ('present semaphore reused', 'open'), ('clear barrier does not chain to the acquire wait', 'open'),
      ('depth image shared across frames, colour-only dependency', 'open')], None),
    (2, 'mpv', 'mpv / libplacebo', '4 configurations', '0.37 / 6.338', ['mpv_gpunext', 'mpv_gpu_hq'],
     [('barrier does not chain to an external semaphore wait', 'up')],
     'frozen: timestamp queries unmodelled (no memory effect), outside the premises'),
    (2, 'gst', 'GStreamer', '5 pipelines', '1.24.2', ['gst_sink_w'],
     [('acquire wait at BOTTOM_OF_PIPE blocks nothing', 'open'), ('sampled image visible to input-attachment reads only', 'open')], None),
]
STATUS = {'up': ('changed upstream before the study', '#5d7f93', '↑'), 'open': ('still present upstream', '#b4532a', '●'),
          'mixed': ('one changed upstream, one still present', '#5d7f93', '↑●')}


def bar(c, label, post=False):
    tot = sum(c.values())
    if tot == 0:
        return (f'<div class="br"><span class="bl">{label}</span><div class="bb empty">no report</div>'
                f'<span class="bn">0</span></div>')
    segs = ''.join(f'<span title="{n}" style="flex:{c.get(k, 0)};background:{col}"></span>' for k, n, col in CLS if c.get(k))
    nums = ' / '.join(f'{c.get(k, 0):,}' for k in ('TRUE', 'TECH', 'FA')) + (f' / {c["UNRESOLVED"]}u' if c.get('UNRESOLVED') else '')
    pct = 100 * c.get('TRUE', 0) / tot
    return (f'<div class="br{" post" if post else ""}"><span class="bl">{label}</span><div class="bb">{segs}</div>'
            f'<span class="bn"><b>{pct:.0f}%</b> {nums}</span></div>')


def card(b):
    st, key, name, progs, rev, imgs, causes, note = b
    im = ''.join(f'<img src="img/{i}.jpg">' for i in imgs)
    fr, fo, rc = (P.get(f'{a}/{m}/{key}', {}) for a, m in (('frozen', 'conservative'), ('frozen', 'observed'), ('release', 'conservative')))
    cs = ''.join(f'<div class="cz"><span class="mk" style="color:{STATUS[s][1]}">{STATUS[s][2]}</span>{t}</div>' for t, s in causes)
    if not causes:
        cs = ''
    nt = f'<div class="nt">{note}</div>' if note else ''
    return (f'<div class="card"><div class="ims n{len(imgs)}">{im}</div><div class="hd"><b>{name}</b> <span>{progs}{' · ' + rev if rev else ''}</span></div>'
            f'{bar(fr, "frozen")}{bar(fo, "observed")}{bar(rc, "release", True)}'
            f'<div class="cl">{len(causes)} true cause{"s" if len(causes) != 1 else ""}</div>{cs}{nt}</div>')


rows = ''
for s, title in ((1, 'Study 1: 32 programs, checker frozen before its first run'),
                 (2, 'Study 2: 25 programs, a later checker frozen before its first run')):
    rows += f'<div class="sh">{title}</div><div class="row">' + ''.join(card(b) for b in BASES if b[0] == s) + '</div>'
legend = (''.join(f'<span class="lg"><span class="sw" style="background:{col}"></span>{n}</span>' for _, n, col in CLS) +
          '<span class="lg">frozen: the fresh result (conservative capture, the theorem\'s configuration) · observed: same checker, observed capture · '
          '<i>release: post-hoc, same traces</i> · counts: true / technicality / false alarm (/ unresolved)</span>'
          '<span class="lg"><span style="color:#5d7f93">↑</span> changed upstream before the study &nbsp;'
          '<span style="color:#b4532a">●</span> still present upstream</span>')
html = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>Fresh studies by code base</title>
<!-- GENERATED by build_precision.py; edit that, not this file. -->
<link rel="stylesheet" href="style.css">
<style>
  #fig {{ width: 672px; font-family: 'Libertinus Sans', sans-serif; }}
  .wrap {{ border: 1px solid var(--panel-border); border-radius: 7px; background: var(--band-bg); padding: 4px 6px 6px; }}
  .sh {{ font-size: 8.4px; font-weight: 700; margin: 3px 1px 3px; color: #2f4656; }}
  .row {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 5px; }}
  .card {{ min-width: 0; background: #fff; border: 1px solid var(--panel-border); border-radius: 4px; padding: 3px 4px 4px; }}
  .ims {{ display: flex; gap: 2px; margin-bottom: 3px; }}
  .ims img {{ flex: 1 1 0; width: 0; min-width: 0; aspect-ratio: 4/3; object-fit: cover; border-radius: 2px; display: block; }}
  .ims.n1 img {{ aspect-ratio: 8/3; }}
  .hd {{ font-size: 8.4px; margin-bottom: 2px; }} .hd span {{ color: var(--muted); font-size: 6.9px; }}
  .br {{ display: grid; grid-template-columns: 30px 1fr 62px; gap: 3px; align-items: center; font-size: 6.6px; margin: 1.2px 0; }}
  .br .bl {{ color: var(--muted); }} .br.post .bl {{ font-style: italic; }}
  .bb {{ display: flex; height: 6.5px; border: 0.6px solid #9aa8b5; border-radius: 1.5px; overflow: hidden; background: #fff; }}
  .br.post .bb {{ border-style: dashed; }}
  .bb.empty {{ font-size: 5.8px; color: var(--ok); align-items: center; padding-left: 3px; border-color: #b9d6c2; }}
  .bn {{ text-align: right; white-space: nowrap; color: var(--ink); }} .bn b {{ font-weight: 700; }}
  .cl {{ font-size: 6.9px; font-weight: 700; margin-top: 3px; color: var(--ink); }}
  .cz {{ font-size: 6.6px; line-height: 1.22; color: var(--ink); padding-left: 8px; text-indent: -8px; }}
  .cz .mk {{ display: inline-block; width: 8px; text-indent: 0; font-size: 6.4px; }}
  .cz.none {{ color: var(--muted); padding-left: 0; text-indent: 0; }}
  .nt {{ font-size: 6.2px; color: var(--muted); font-style: italic; margin-top: 2px; line-height: 1.22; }}
  .legend {{ margin-top: 5px; font-size: 6.9px; color: var(--ink); display: flex; flex-wrap: wrap; gap: 2px 9px; }}
  .sw {{ display: inline-block; width: 8px; height: 6.5px; border: 0.6px solid #9aa8b5; vertical-align: -0.5px; margin-right: 2px; }}
</style></head><body><div id="fig"><div class="wrap">{rows}<div class="legend">{legend}</div></div>
<div class="caption" id="cap">Fig. (preview)</div></div></body></html>'''
(HERE / 'fig-precision.html').write_text(html)
tot = {k: sum(P.get(f'frozen/conservative/{b[1]}', {}).get(k, 0) for b in BASES if b[0] == 1) for k in ('TRUE', 'TECH', 'FA')}
assert tot == {'TRUE': 130, 'TECH': 12, 'FA': 1971}, tot
print('wrote fig-precision.html')
