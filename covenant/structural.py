"""Structural checks: RCGP's checks, re-expressed over the Contract IR.

* shader I/O between stages, matched by NAME (RCGP matches by position,
  which its authors note causes bugs when outputs are reordered, fn. 6)
* host struct layout vs the contract's std430 layout (RCGP Issue 3)
* command order: every resource a pipeline imports must be bound before a
  draw or dispatch (RCGP Issue 5, its Dependency/Resolvant/Sentinel algebra)
"""
from .diagnostics import Report

SIZE = {"float": 4, "int": 4, "vec2": 8, "vec3": 12, "vec4": 16, "mat4": 64}
STD430_ALIGN = {"float": 4, "int": 4, "vec2": 8, "vec3": 16, "vec4": 16, "mat4": 16}
C_ALIGN = {"float": 4, "int": 4, "vec2": 4, "vec3": 4, "vec4": 4, "mat4": 4}


def _layout(fields, align):
    off, out = 0, []
    for name, ty in fields:
        a = align[ty]
        off = (off + a - 1) // a * a
        out.append((name, ty, off))
        off += SIZE[ty]
    return out


def std430_offsets(contract):
    return _layout([(f.name, f.ty) for f in contract.fields], STD430_ALIGN)


def check_host_layout(contract, host_fields, where, report=None):
    """host_fields: [(name, ty)] of a hand-written C++ struct (natural C
    alignment, as with glm types)."""
    report = report or Report()
    dev = std430_offsets(contract)
    host = _layout(host_fields, C_ALIGN)
    for (dn, dt, do), (hn, ht, ho) in zip(dev, host):
        if dn != hn or dt != ht:
            report.add("structure", "structure.host-field",
                       f"host struct field '{hn}: {ht}' does not match "
                       f"'{dn}: {dt}' in '{contract.name}'.", where)
        elif do != ho:
            report.add("structure", "structure.host-layout",
                       f"field '{dn}' is at byte {ho} on the host but at byte {do} "
                       f"in '{contract.name}' ({contract.layout}); the GPU reads "
                       f"shifted data.", where,
                       ["use the synthesized host type for this contract instead "
                        "of a hand-written replica"])
    return report


def check_io(pipeline, report=None, positional=False):
    report = report or Report()
    for up, down in zip(pipeline.stages, pipeline.stages[1:]):
        outs, ins = list(up.outputs.items()), list(down.inputs.items())
        if positional:                              # RCGP behaviour
            pairs = [(o, i) for o, i in zip(outs, ins)]
        else:
            pairs = [((n, up.outputs[n]), (n, e)) if n in up.outputs else (None, (n, e))
                     for n, e in ins]
        for o, i in pairs:
            if o is None:
                report.add("structure", "structure.io-missing",
                           f"'{down.name}' reads varying '{i[0]}' but '{up.name}' "
                           f"never writes it.", f"{pipeline.name}")
            elif o[1].ty != i[1].ty:
                report.add("structure", "structure.io-type",
                           f"varying '{i[0]}' is {o[1].ty} in '{up.name}' "
                           f"but {i[1].ty} in '{down.name}'.", f"{pipeline.name}")
            # positional mode deliberately reports nothing when two same-typed
            # varyings are swapped: that is the silent bug RCGP admits.
    return report


def positional_pairs(pipeline):
    """How RCGP would wire varyings: (output name, input name) by position."""
    up, down = pipeline.stages[0], pipeline.stages[1]
    return list(zip(up.outputs, down.inputs))


def check_commands(commands, report=None):
    """commands: list of tuples ("bind_pipeline", Pipeline) | ("bind", Contract)
    | ("draw",) | ("dispatch",). Mirrors RCGP's sentinel rule."""
    report = report or Report()
    need, have = set(), set()
    for i, cmd in enumerate(commands):
        if cmd[0] == "bind_pipeline":
            need = {c.name for c in cmd[1].imports
                    if c.kind not in ("render_target", "attribute_stream")}
            have = set()
            pipe = cmd[1]
        elif cmd[0] == "bind":
            have.add(cmd[1].name)
        elif cmd[0] in ("draw", "dispatch"):
            missing = sorted(need - have)
            if missing:
                report.add("structure", "structure.missing-binding",
                           f"{cmd[0]} uses '{pipe.name}' before these are bound: "
                           f"{', '.join(missing)}.", f"command {i}")
    return report
