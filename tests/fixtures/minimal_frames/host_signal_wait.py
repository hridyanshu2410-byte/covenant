import sys
sys.path.insert(0, sys.argv[1])
from covenant.timing import Frame, Use, Signal, Wait, check
from covenant import timing_ref
F=frozenset
fr = Frame()
fr.q("host").extend([Signal("submit@1"), Use("buf","write",stage="HOST",flag="HOST_WRITE"), Signal("t=5")])
fr.q("graphics0").extend([Wait("submit@1"), Wait("t=5", stages=F({"FRAGMENT_SHADER"})),
                          Use("buf","read",stage="VERTEX_SHADER",flag="UNIFORM_READ")])
for ex in (True, False):
    hz=[]; check(fr, hazards=hz, exhaustive=ex, window=False, discard_relaxed=False); print("pairwise" if ex else "epoch", hz)
print("ref", timing_ref.race_keys(fr))
