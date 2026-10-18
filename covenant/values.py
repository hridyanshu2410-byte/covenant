"""Concrete values a draw's shaders read, replayed from the model driver's
trace records (host_trace.HostModel.records):

* push constants: vkCmdPushConstants bytes in effect at each draw;
* uniform / storage buffer bytes: the descriptor's buffer range, rebuilt from
  host_write records that carry byte contents ("writes": [[offset, hex]],
  model driver >= 2) up to the draw's submission;
* readback fills and image-I/O sink calls (covenant_readback_fill and
  covenant_io records) for the dynamic host checks in witness.py.

All of this is best effort: a value whose bytes are not fully known is left
out, and the probe then treats it as an unknown parameter.
"""
from collections import defaultdict


def _hex(s):
    try:
        return bytes.fromhex(s)
    except (TypeError, ValueError):
        return None


class Values:
    def __init__(self, model):
        self.model = model
        self.push = {}            # draw seq -> (bytes, mask)
        self.desc = {}            # draw seq -> {(set, binding): (buffer, offset, range)}
        self.buf_mem = {}         # buffer -> (memory, offset, size)
        self.mem_writes = defaultdict(list)    # memory -> [(seq, offset, bytes)]
        self.readback_fills = []  # records
        self.io = []              # covenant_io records
        self.truncated = set()    # memories whose content records were cut short
        self._replay(model.records)

    # ------------------------------------------------------------ replay
    def _replay(self, recs):
        sets = defaultdict(dict)                  # descriptor set -> binding -> (buffer, off, range)
        pc = {}                                   # cb -> (bytearray, bytearray mask)
        bound = defaultdict(dict)                 # cb -> set index -> descriptor set
        for r in recs:
            fn = r.get("fn", "")
            if fn == "vkBindBufferMemory":
                self.buf_mem[r.get("buffer")] = (r.get("memory"), r.get("offset", 0), r.get("size"))
            elif fn in ("vkBindBufferMemory2", "vkBindBufferMemory2KHR"):
                for b in r.get("binds", []):
                    self.buf_mem[b.get("buffer")] = (b.get("memory"), b.get("offset", 0), b.get("size"))
            elif fn == "vkUpdateDescriptorSets":
                for w in r.get("writes", []):
                    if "BUFFER" not in str(w.get("type", "")):
                        continue
                    for i, el in enumerate(w.get("elements", [])):
                        if i == 0 and "buffer" in el:
                            sets[w.get("set")][w.get("binding")] = (el["buffer"], el.get("offset", 0),
                                                                     el.get("range"))
            elif fn in ("vkBeginCommandBuffer", "vkResetCommandBuffer"):
                pc.pop(r.get("cb"), None)
                bound.pop(r.get("cb"), None)
            elif fn == "vkCmdPushConstants":
                data = _hex(r.get("data", ""))
                if data is None:
                    continue
                buf, mask = pc.setdefault(r.get("cb"), (bytearray(256), bytearray(256)))
                off = int(r.get("offset", 0))
                if off + len(data) > len(buf):
                    grow = off + len(data) - len(buf)
                    buf.extend(bytes(grow))
                    mask.extend(bytes(grow))
                buf[off:off + len(data)] = data
                mask[off:off + len(data)] = b"\x01" * len(data)
            elif fn == "vkCmdBindDescriptorSets":
                for i, ds in enumerate(r.get("sets", [])):
                    bound[r.get("cb")][r.get("first_set", 0) + i] = ds
            elif fn.startswith(("vkCmdDraw", "vkCmdDispatch")):
                cb = r.get("cb")
                if cb in pc:
                    self.push[r["seq"]] = (bytes(pc[cb][0]), bytes(pc[cb][1]))
                d = {}
                for si, ds in bound.get(cb, {}).items():
                    for b, v in sets.get(ds, {}).items():
                        d[(si, b)] = v
                self.desc[r["seq"]] = d
            elif fn == "host_write" and "writes" in r:
                if r.get("writes_truncated") or r.get("content_truncated"):
                    self.truncated.add(r.get("memory"))
                self.mem_writes[r.get("memory")]
                for w in r["writes"]:
                    off, hx = w[0], w[1]
                    if not hx:
                        continue            # older driver: length-only run (no content recorded)
                    data = _hex(hx)
                    if data is not None:
                        self.mem_writes[r.get("memory")].append((r["seq"], int(off), data))
            elif fn == "covenant_readback_fill":
                self.readback_fills.append(r)
            elif fn == "covenant_io":
                self.io.append(r)

    def load_io(self, path):
        """Records of the image-I/O interposer (model_driver/covio.c)."""
        import json
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.io.append(json.loads(line))
                        except ValueError:
                            pass
        except OSError:
            pass

    # ------------------------------------------------------------ queries
    def memory_bytes(self, memory, offset, size, before_seq):
        """(bytes, mask) of memory[offset:offset+size] as the host left it
        before before_seq, or None when nothing is known. The driver's mapped
        memory starts zeroed and host_write records carry every changed byte,
        so once a memory has content records all of it is known (unless a
        record was truncated)."""
        if size is None or size <= 0 or size > (1 << 24):
            return None
        recs = self.mem_writes.get(memory)
        if not recs or memory in self.truncated:
            return None
        buf = bytearray(size)
        for seq, off, data in recs:
            if before_seq is not None and seq >= before_seq:
                break
            lo, hi = max(off, offset), min(off + len(data), offset + size)
            if lo < hi:
                buf[lo - offset:hi - offset] = data[lo - off:hi - off]
        return bytes(buf), b"\x01" * size

    def buffer_bytes(self, buffer, offset, rng, before_seq):
        bm = self.buf_mem.get(buffer)
        if bm is None:
            return None
        mem, moff, msize = bm
        info = self.model.buffers.get(buffer) if hasattr(self.model, "buffers") else None
        size = getattr(info, "size", None) or msize
        if rng in (None, "whole") or not isinstance(rng, int):
            rng = (size or 0) - (offset or 0)
        return self.memory_bytes(mem, (moff or 0) + (offset or 0), rng, before_seq)

    def for_draw(self, draw):
        """{"push": (bytes, mask), (set, binding): (bytes, mask)} for a DrawEvent."""
        out = {}
        if draw.seq in self.push:
            out["push"] = self.push[draw.seq]
        for key, (buf, off, rng) in self.desc.get(draw.seq, {}).items():
            got = self.buffer_bytes(buf, off, rng, getattr(draw, "submit_seq", None))
            if got is not None:
                out[key] = got
        return out
