#!/usr/bin/env python3
"""Compare the shader access classification (spirv.entry_binding_modes, with and
without byte spans) of two checker source trees on every SPIR-V module under the
given directories. A change confined to that function leaves every check result
unchanged on a corpus where this reports no differing module.

usage: shader_modes_diff.py <old checker root> <new checker root> <dir> [<dir> ...]
(a checker root is a directory that contains covenant/; extract the corpora's
trace archives under experiments/*/data first)"""
import glob
import hashlib
import importlib
import os
import sys


def load(root):
    for k in [k for k in sys.modules if k == "covenant" or k.startswith("covenant.")]:
        del sys.modules[k]
    sys.path.insert(0, root)
    try:
        return importlib.import_module("covenant.spirv")
    finally:
        sys.path.pop(0)


def classify(sp, modules):
    out = {}
    for h, path in modules.items():
        m = sp.parse(path)
        r = {}
        for _, _, name, _ in m.entries:
            for spans in (False, True):
                try:
                    r[(name, spans)] = sp.entry_binding_modes(m, name, spans=spans)
                except Exception as e:      # an incomplete analysis must stay incomplete
                    r[(name, spans)] = ("error", type(e).__name__)
        out[h] = r
    return out


def main():
    old, new, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    modules = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.spv"), recursive=True)):
            if os.path.isfile(f):
                with open(f, "rb") as fh:
                    modules.setdefault(hashlib.sha1(fh.read()).hexdigest(), f)
    a, b = classify(load(old), modules), classify(load(new), modules)
    diff = sorted(modules[h] for h in modules if a[h] != b[h])
    print(f"{len(modules)} distinct modules, {len(diff)} classified differently")
    for f in diff:
        print("  " + f)
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
