"""Capture a frame of each fresh-study program in caplist.txt on lavapipe (Figure 7 and the teaser).
usage: FRESH=<fresh-study build dir> python3 caplist.py <outdir> [program ...]
The mpv lines play $FRESH/testsrc.mp4, made with
  ffmpeg -f lavfi -i testsrc=size=640x360:rate=30 -t 20 -pix_fmt yuv420p $FRESH/testsrc.mp4
libplacebo refuses a software device unless it is named, hence --vulkan-device."""
import os, shlex, subprocess, sys
W = os.path.dirname(os.path.abspath(__file__)); out = sys.argv[1]; only = sys.argv[2:]
os.makedirs(out, exist_ok=True)
for line in open(os.path.join(W, 'caplist.txt')):
    pid, wd, envs, cmd, delay = os.path.expandvars(line.rstrip('\n')).split('|')
    if only and pid not in only: continue
    env = dict(os.environ)
    if envs != '-':
        for kv in envs.split(): k, v = kv.split('=', 1); env[k] = v
    subprocess.run(['python3', os.path.join(W, 'cap.py'), os.path.join(out, pid + '.png'), '1280x720', delay, '--'] + shlex.split(cmd), cwd=wd, env=env)
