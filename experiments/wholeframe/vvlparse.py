"""Parse a Khronos validation-layer log (log_filename output) into messages."""
import re

HDR = re.compile(r"^Validation (Error|Warning|Performance Warning|Information): \[ ([^\]]+) \] \| MessageID = (0x[0-9a-f]+)", re.M)
_NM = r"\[((?:[^\[\]]|\[[^\[\]]*\])*)\]"      # a debug name, which may itself contain [i]
RES = re.compile(r"Vk(Image|Buffer|ImageView) 0x[0-9a-fA-F]+" + _NM)
NAMED = re.compile(r"(Vk[A-Za-z]+) 0x[0-9a-fA-F]+" + _NM)


def parse(text):
    out = []
    hs = list(HDR.finditer(text))
    for i, h in enumerate(hs):
        body = text[h.end(): hs[i + 1].start() if i + 1 < len(hs) else len(text)]
        main, _, objs = body.partition("\nObjects:")
        main1 = main.split("The Vulkan spec states:")[0]
        res = None
        m = RES.search(main1)
        if m:
            res = m.group(2)
            if m.group(1) == "ImageView" and res.endswith(".view"):
                res = res[:-5]
        named = []
        for k, n in NAMED.findall(main1) + NAMED.findall(objs):
            if (k, n) not in named:
                named.append((k, n))
        call = re.search(r"\b(vk[A-Za-z0-9]+)\(\)", main1)
        out.append({"id": h.group(2), "sev": h.group(1), "res": res, "named": named[:8],
                    "call": call.group(1) if call else None, "text": main1.strip()[:600]})
    return out
