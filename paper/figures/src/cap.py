#!/usr/bin/env python3
"""Run a program on lavapipe under Xvfb and grab the screen after a delay.
usage: cap.py OUT.png WxH DELAY -- cmd args...   (env passed through; DISPLAY set here)"""
import os, re, subprocess, sys, time, signal

out, size, delay = sys.argv[1], sys.argv[2], float(sys.argv[3])
cmd = sys.argv[sys.argv.index('--') + 1:]
disp = ':%d' % (90 + (os.getpid() % 50))
xv = subprocess.Popen(['Xvfb', disp, '-screen', '0', size + 'x24', '-nolisten', 'tcp'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.0)
env = dict(os.environ, DISPLAY=disp,
           VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.json',
           VK_DRIVER_FILES='/usr/share/vulkan/icd.d/lvp_icd.json')
env.setdefault('GDK_BACKEND', 'x11')
env.setdefault('SDL_VIDEO_DRIVER', 'x11')
log = open(out + '.log', 'w')
app = subprocess.Popen(cmd, env=env, stdout=log, stderr=log, start_new_session=True)
time.sleep(delay)
rc = app.poll()
# crop to the largest mapped top-level window
best = None
tree = subprocess.run(['xwininfo', '-display', disp, '-root', '-children'], capture_output=True, text=True).stdout
for wid in re.findall(r'^\s+(0x[0-9a-f]+)', tree, re.M):
    info = subprocess.run(['xwininfo', '-display', disp, '-id', wid], capture_output=True, text=True).stdout
    if 'IsViewable' not in info:
        continue
    g = {k: int(re.search(k + r':\s+(-?\d+)', info).group(1)) for k in ('Absolute upper-left X', 'Absolute upper-left Y', 'Width', 'Height')}
    a = g['Width'] * g['Height']
    if best is None or a > best[0]:
        best = (a, g)
subprocess.run(['import', '-display', disp, '-window', 'root', out], check=False)
if best:
    g = best[1]
    subprocess.run(['convert', out, '-crop', '%dx%d+%d+%d' % (g['Width'], g['Height'], max(0, g['Absolute upper-left X']), max(0, g['Absolute upper-left Y'])), '+repage', out])
for p in (app,):
    try:
        os.killpg(app.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
try:
    app.wait(5)
except subprocess.TimeoutExpired:
    os.killpg(app.pid, signal.SIGKILL)
xv.terminate(); xv.wait()
print(out, 'app exited early rc=%s' % rc if rc is not None else 'ok')
