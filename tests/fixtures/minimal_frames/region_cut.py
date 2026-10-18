import sys
sys.path.insert(0, sys.argv[1])
from covenant.timing import Frame, Use, Barrier, check
from covenant import timing_ref
F=frozenset
fr = Frame()
fr.q("q").extend([
    Use("b", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE", region=((0,100),)),
    Barrier(None, src_stages=F({"COMPUTE_SHADER"}), src_access=F({"SHADER_STORAGE_WRITE"}), dst_stages=F({"COMPUTE_SHADER"}), dst_access=F({"SHADER_STORAGE_WRITE"})),
    Use("b", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE", region=((0,50),)),
    Use("b", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE", region=((50,100),)),
    Barrier("b", src_stages=F({"COMPUTE_SHADER"}), src_access=F({"SHADER_STORAGE_WRITE"}), dst_stages=F({"FRAGMENT_SHADER"}), dst_access=F({"SHADER_STORAGE_READ"}), region=((0,50),), call="c", struct=0),
    Barrier("b", src_stages=F({"COMPUTE_SHADER"}), src_access=F({"SHADER_STORAGE_WRITE"}), dst_stages=F({"FRAGMENT_SHADER"}), dst_access=F({"SHADER_STORAGE_READ"}), region=((50,100),), call="c", struct=1),
    Use("b", "read", stage="FRAGMENT_SHADER", flag="SHADER_STORAGE_READ", region=((0,100),)),
])
for ex in (True, False):
    hz=[]; check(fr, hazards=hz, exhaustive=ex, window=False, discard_relaxed=False); print("pairwise" if ex else "epoch", hz)
print("ref", timing_ref.race_keys(fr))
