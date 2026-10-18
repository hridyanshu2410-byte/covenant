"""Plain-sentence diagnostics shared by every checker."""
from dataclasses import dataclass, field


@dataclass
class Diagnostic:
    checker: str          # "structure" | "meaning" | "timing"
    code: str             # short stable id, e.g. "meaning.color-mismatch"
    message: str          # one plain sentence a person can act on
    where: str = ""       # module / command location
    notes: list = field(default_factory=list)  # provenance lines
    severity: str = "error"   # "error" | "warning" (relies on a heuristic)

    def __str__(self):
        sev = "" if self.severity == "error" else f" ({self.severity})"
        head = f"[{self.checker}] {self.code}{sev}: {self.message}"
        if self.where:
            head += f"\n    at {self.where}"
        for n in self.notes:
            head += f"\n    note: {n}"
        return head


class Report:
    def __init__(self):
        self.items = []
        self._keys = set()
        self.meta = {}          # facts about the check that are not diagnostics (e.g. "timing.window")

    def add(self, *args, **kw):
        d = Diagnostic(*args, **kw)
        k = (d.code, d.message, d.where)
        if len(self._keys) != len(self.items):        # items edited from outside
            self._keys = {(x.code, x.message, x.where) for x in self.items}
        if k not in self._keys:
            self._keys.add(k)
            self.items.append(d)

    def extend(self, other):
        self.items.extend(other.items)

    @property
    def ok(self):
        return not self.items

    @property
    def errors(self):
        return [d for d in self.items if d.severity == "error"]

    def codes(self):
        return [d.code for d in self.items]

    def __str__(self):
        if self.ok:
            return "all checks passed"
        return "\n".join(str(d) for d in self.items)
