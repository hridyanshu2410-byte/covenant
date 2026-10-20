"""Derive the figure data from the archived experiment results.

precision.json (Figure 7): report groups per code base and class, from the triaged groups of
  experiments/fresh-studies/data/classes-{frozen,release}-study{1,2}.json; gtk_path_walk is left out
  (not grouped when frozen, grouping incomplete in the release).
repairs.json (Figures 1 and 8): per repaired program, the timing diagnostics of the original run by
  cause and of the patched run, from the archived checks in experiments/repairs/data/traces.tar.xz.
Run: python3 prep_data.py [--check]   (--check: compare with the JSON files instead of writing them)"""
import collections
import importlib.util
import json
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / "experiments"
BASE = {"q3d": "qt", "qq": "qt", "wgpu": "wgpu", "imgui": "imgui", "vkcube": "vkcube", "gtk": "gtk",
        "vkmark": "vkmark", "mpv": "mpv", "gst": "gst"}


def precision():
    out = collections.defaultdict(collections.Counter)
    progs = collections.Counter()
    for stage in ("frozen", "release"):
        for study in (1, 2):
            data = json.loads((EXP / "fresh-studies" / "data" / f"classes-{stage}-study{study}.json").read_text())
            for key, groups in data.items():
                prog, mode = key.split("/")
                if prog == "gtk_path_walk":
                    continue
                base = "vkcube" if prog == "vkcube" else BASE[prog.split("_")[0]]
                c = out[f"{stage}/{mode}/{base}"]
                c.update(g["class"] for g in groups)
                c.update()                      # a program without reports still makes the key
                if (stage, mode) == ("frozen", "conservative"):
                    progs[base] += 1
    res = {k: dict(sorted(v.items())) for k, v in sorted(out.items())}
    res["progs"] = dict(sorted(progs.items()))
    return res


REPAIRS = [  # (project, original variant, patched variant, case, programs)
    ("vkmark", "base", "all", "VKMARK-ALL", ["clear", "cube", "desktop", "effect2d", "shading", "texture", "vertex"]),
    ("gst", "orig", "both", "GST-ALL", ["sink", "colorconvert", "overlay", "viewconvert", "shaderspv"]),
    ("imgui", "orig", "fix", "IMGUI-PRE", ["glfw", "sdl2"]),
    ("sdl", "orig", "fix", "SDL-1", ["testsprite"]),
]


def repairs():
    spec = importlib.util.spec_from_file_location("make_case", EXP / "repairs" / "scripts" / "make_case.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)                 # its attribution rules (cause of each diagnostic)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(EXP / "repairs" / "data" / "traces.tar.xz") as tf:
            tf.extractall(tmp, filter="data")
        for proj, orig, fix, case, progs in REPAIRS:
            for p in progs:
                prog = f"{proj}_{p}"
                runs = {v: json.loads((Path(tmp) / "runs" / proj / v / prog / "conservative" / "check.json").read_text())
                        ["findings"] for v in (orig, fix)}
                timing = {v: [f for f in fs if f["code"].startswith("timing.")] for v, fs in runs.items()}
                causes = collections.Counter(mc.cause(prog, f) for f in timing[orig])
                out[prog] = {"case": case, "before": len(timing[orig]), "after": len(timing[fix]),
                             "causes": dict(sorted(causes.items()))}
    return out


def main():
    new = {"precision.json": precision(), "repairs.json": repairs()}
    for name, data in new.items():
        path = HERE / name
        if "--check" in sys.argv:
            old = json.loads(path.read_text())
            print(name, "matches" if old == data else "DIFFERS")
            if old != data:
                for k in sorted(set(old) | set(data)):
                    if old.get(k) != data.get(k):
                        print(" ", k, old.get(k), data.get(k))
        else:
            path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
