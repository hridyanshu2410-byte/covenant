"""Run the forward renderer clean and with each seeded bug; print what each
checker reports and what RCGP's checks would have caught."""
import sys
sys.path.insert(0, "examples")
from covenant import check_all, structural
from forward import build

BUGS = ["srgb", "world_view", "double_encode", "units_cm", "host_layout",
        "missing_bind", "async_race", "in_flight", "no_barrier"]

def run(bugs):
    prog = build(bugs)
    return check_all(prog["pipelines"], prog["host_writes"], prog["frame"],
                     prog["commands"], prog["layouts"])

if __name__ == "__main__":
    r = run(())
    print("clean program:", r)
    for b in BUGS:
        print(f"\n=== bug: {b}")
        print(run([b]))
