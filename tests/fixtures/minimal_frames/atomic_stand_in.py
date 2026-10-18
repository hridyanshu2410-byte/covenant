import sys
sys.path.insert(0, sys.argv[1])
from covenant.timing import Frame, Use, Barrier, check
from covenant import timing_ref
A = frozenset({("atomic", None)})
fr = Frame()
fr.q("graphics0").extend([
    Use("counter", "write", stage="CLEAR", flag="TRANSFER_WRITE", cmd="fill"),
    Barrier(None, src_stages=frozenset({"TRANSFER"}), src_access=frozenset({"TRANSFER_WRITE"}),
            dst_stages=frozenset({"COMPUTE_SHADER"}), dst_access=frozenset({"SHADER_STORAGE_READ", "SHADER_STORAGE_WRITE"})),
    Use("counter", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE", cmd="dispatch", atomic=A),
    Use("counter", "write", stage="FRAGMENT_SHADER", flag="SHADER_STORAGE_WRITE", cmd="draw", atomic=A),
])
for ex in (True, False):
    hz = []; check(fr, hazards=hz, exhaustive=ex, window=False, discard_relaxed=False)
    print("exhaustive" if ex else "epoch", hz)
print("ref", timing_ref.race_keys(fr))
