"""Meaning signatures of library and API entry points.

Each module states, from the *specification* of an API (never from a bug),
which meaning dimension a call or a piece of fixed-function state constrains
and with which group element. The labelled solver (labels.py) turns these into
edges; a conflict is a cycle with a non-identity product.

Element conventions (see labels.py):
  color       FreeGroup word in application order, (("srgb", +1),) = sRGB encode
  conv.alpha  Z: number of premultiplications (0 = straight)
  alpha.power Z: power of coverage stored in an alpha channel (1 = coverage)
  clip_y      Z2: 0 = clip-space +Y up (OpenGL / GLM), 1 = +Y down (Vulkan)
  depth.range Z: 0 = [0, 1] clip depth, 1 = [-1, 1]; 2x-1 is +1, 0.5x+0.5 is -1
  depth.dir   Z2: 0 = forward (nearer is smaller), 1 = reversed
  angle       Z: power of pi/180 (0 = radians, 1 = degrees)
  unit        UnitGroup element (labels.UnitGroup.of)

Modules: vulkan (formats, color spaces, blend, viewport, compare ops, depth
clip control), glsl_std450 (extended instructions), glm (projections, lookAt,
radians/degrees, inverse), std_trig (C/C++ <cmath>), image_io (stb_image,
stb_image_write, libpng, lodepng, tinyexr).
"""
from . import glm, glsl_std450, image_io, projection, std_trig, vulkan  # noqa: F401

SRGB_ENC = (("srgb", 1),)
SRGB_DEC = (("srgb", -1),)
PQ_ENC = (("pq", 1),)
HLG_ENC = (("hlg", 1),)

# encoding names used by annotations ([Color("srgb")]) -> element from linear
ENCODING = {"linear": (), "srgb": SRGB_ENC, "pq": PQ_ENC, "hlg": HLG_ENC}


def encoding_element(name):
    """Element of a named encoding relative to linear light. Unknown names
    (e.g. 'rgbm') become their own generator; 'data' / 'any' are wildcards
    (None)."""
    if name in ("data", "any"):
        return None
    return ENCODING.get(name, ((name, 1),))
