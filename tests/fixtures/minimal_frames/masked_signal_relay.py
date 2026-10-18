import sys
sys.path.insert(0, sys.argv[1])
from covenant.timing import Frame, Use, Barrier, Signal, Wait, check
from covenant import timing_ref
from covenant.diagnostics import Report

def run(name, fr):
    hz = []
    check(fr, hazards=hz, exhaustive=True, window=False)
    hz2 = []
    check(fr, hazards=hz2, exhaustive=False, window=False)
    ref = timing_ref.race_keys(fr)
    print(name, "pairwise:", [(a,b,c) for a,b,c in hz], "epoch:", len(hz2), "ref:", ref)

# 1. relay through a third queue whose signal stage mask does not meet its wait's stage mask
fr = Frame()
fr.q("compute0").extend([Use("buf", "write", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_WRITE"),
                         Signal("a")])
fr.q("graphics0").extend([Wait("a", stages=frozenset({"VERTEX_SHADER"})),
                          Signal("b", stages=frozenset({"COMPUTE_SHADER"}))])
fr.q("transfer0").extend([Wait("b"), Use("buf", "read", stage="COPY", flag="TRANSFER_READ")])
run("relay-queue", fr)

# 1b. same but direct hop (control): signal with stage mask not covering the write
fr = Frame()
fr.q("compute0").extend([Use("buf", "write", stage="FRAGMENT_SHADER", flag="SHADER_STORAGE_WRITE"),
                         Signal("a", stages=frozenset({"COMPUTE_SHADER"}))])
fr.q("transfer0").extend([Wait("a"), Use("buf", "read", stage="COPY", flag="TRANSFER_READ")])
run("direct", fr)

# 2. relay through the host: masked signal waited by host, host submits to another queue
fr = Frame()
fr.q("graphics0").extend([Use("buf", "write", stage="FRAGMENT_SHADER", flag="SHADER_STORAGE_WRITE"),
                          Signal("t=1", stages=frozenset({"COMPUTE_SHADER"}))])
fr.q("host").extend([Wait("t=1"), Signal("submit@9")])
fr.q("compute0").extend([Wait("submit@9"), Use("buf", "read", stage="COMPUTE_SHADER", flag="SHADER_STORAGE_READ")])
run("host-relay", fr)

# 3. minimal UNDEFINED test: write, execution-only dependency, transition from UNDEFINED
for relaxed in (True, False):
    fr = Frame()
    fr.q("graphics0").extend([
        Use("img", "write", stage="COLOR_ATTACHMENT_OUTPUT", flag="COLOR_ATTACHMENT_WRITE"),
        Barrier("img", None, "transfer_dst", src_stages=frozenset({"COLOR_ATTACHMENT_OUTPUT"}), src_access=frozenset(),
                dst_stages=frozenset({"COPY"}), dst_access=frozenset({"TRANSFER_WRITE"}))])
    hz = []
    check(fr, hazards=hz, exhaustive=True, window=False, discard_relaxed=relaxed)
    print("undefined relaxed=%s" % relaxed, hz, timing_ref.race_keys(fr, discard_relaxed=relaxed))
