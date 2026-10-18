"""Vulkan fixed-function meaning signatures.

Sources: Vulkan 1.3 specification (VkFormat "Identification of Formats",
"sRGB Texture Conversions" (16.3.2 / 3.9), "Blending" (28.1), "Controlling
the Viewport" (27.9), "Depth Comparison" / VkCompareOp (28.x), "Primitive
Clipping" incl. VK_EXT_depth_clip_control), VK_KHR_surface / VK_EXT_swapchain
_colorspace (VkColorSpaceKHR), VK_KHR_maintenance1 (negative viewport height).
"""

# --- formats ---------------------------------------------------------------
# A *_SRGB format: "the R, G and B components are ... sRGB nonlinear encoded";
# reads decode to linear (sRGB EOTF), writes encode (inverse EOTF). Alpha is
# linear.
def format_transfer(fmt):
    """'srgb' when the format encodes / decodes sRGB in hardware, else None."""
    return "srgb" if fmt and "SRGB" in str(fmt).upper() else None


def is_depth_format(fmt):
    f = str(fmt or "").upper().replace("VK_FORMAT_", "")
    return f.startswith(("D16", "D24", "D32", "X8_D24"))


# Depth attachment values are always in [0, 1] after the viewport transform
# (zf = pz * z + oz with minDepth / maxDepth in [0, 1]); sampling a depth view
# returns that value.
DEPTH_TEXEL_RANGE = 0          # depth.range element of a sampled depth value


# --- swapchain color spaces --------------------------------------------------
# VkColorSpaceKHR -> encoding the presentation engine expects in the image
# (VK_EXT_swapchain_colorspace "Color Spaces"). PASS_THROUGH and unknown
# spaces say nothing. *_NONLINEAR spaces use the sRGB-style transfer.
COLOR_SPACE_ENCODING = {
    "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR": "srgb",
    "VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT": "srgb",
    "VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT": "srgb",
    "VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT": "srgb",
    "VK_COLOR_SPACE_BT709_NONLINEAR_EXT": "srgb",
    "VK_COLOR_SPACE_HDR10_ST2084_EXT": "pq",
    "VK_COLOR_SPACE_DOLBYVISION_EXT": "pq",
    "VK_COLOR_SPACE_HDR10_HLG_EXT": "hlg",
    "VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT": "linear",
    "VK_COLOR_SPACE_BT709_LINEAR_EXT": "linear",
    "VK_COLOR_SPACE_BT2020_LINEAR_EXT": "linear",
    "VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT": "linear",
}
# primaries the color space implies (for color-matrix witnesses)
COLOR_SPACE_PRIMARIES = {
    "VK_COLOR_SPACE_HDR10_ST2084_EXT": "bt2020", "VK_COLOR_SPACE_HDR10_HLG_EXT": "bt2020",
    "VK_COLOR_SPACE_BT2020_LINEAR_EXT": "bt2020", "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR": "bt709",
    "VK_COLOR_SPACE_BT709_LINEAR_EXT": "bt709", "VK_COLOR_SPACE_BT709_NONLINEAR_EXT": "bt709",
}

# --- blending ----------------------------------------------------------------
BF = "VK_BLEND_FACTOR_"
OVER_STRAIGHT = (BF + "SRC_ALPHA", BF + "ONE_MINUS_SRC_ALPHA")
OVER_PREMUL = (BF + "ONE", BF + "ONE_MINUS_SRC_ALPHA")
ADD = "VK_BLEND_OP_ADD"

# color factors (src, dst) with VK_BLEND_OP_ADD -> (conv.alpha count the
# source color must have, premultiplications the blend applies to it):
#   SRC_ALPHA, ONE_MINUS_SRC_ALPHA: Cs * As + Cd * (1 - As): source straight,
#       the blend multiplies by As once (the stored color is premultiplied)
#   ONE, ONE_MINUS_SRC_ALPHA: Cs + Cd * (1 - As): source premultiplied
COLOR_BLEND = {OVER_STRAIGHT: (0, 1), OVER_PREMUL: (1, 0)}
# alpha factors (src, dst) -> power of coverage added to the stored alpha:
#   SRC_ALPHA, ONE_MINUS_SRC_ALPHA: As * As + Ad * (1 - As): alpha squared
#   ONE, ONE_MINUS_SRC_ALPHA:      As + Ad * (1 - As): coverage (Porter-Duff over)
ALPHA_BLEND = {OVER_STRAIGHT: 1, OVER_PREMUL: 0}


def color_blend(bl):
    """(required source count, applied premultiplications) or None."""
    if not bl or not bl.get("enable") or bl.get("color_op", ADD) != ADD:
        return None
    return COLOR_BLEND.get((bl.get("src_color"), bl.get("dst_color")))


def alpha_blend(bl):
    if not bl or not bl.get("enable") or bl.get("alpha_op", ADD) != ADD:
        return None
    return ALPHA_BLEND.get((bl.get("src_alpha"), bl.get("dst_alpha")))


# --- viewport ------------------------------------------------------------------
# VK_KHR_maintenance1: a negative viewport height flips Y. It exists to take
# clip coordinates in the OpenGL convention (+Y up), so it asserts that the
# clip position is +Y up (clip_y 0) and flips it (+1) into framebuffer +Y down.
def viewport_flips(vp):
    return vp is not None and len(vp) >= 4 and vp[3] < 0


# Vulkan framebuffer / clip convention: +Y down (clip_y 1).
CLIP_Y_API = 1

# --- depth ------------------------------------------------------------------------
# VkCompareOp: with LESS / LESS_OR_EQUAL the nearer fragment wins when nearer
# means smaller depth (forward); GREATER / GREATER_OR_EQUAL mean reversed.
COMPARE_DIR = {"VK_COMPARE_OP_LESS": 0, "VK_COMPARE_OP_LESS_OR_EQUAL": 0,
               "VK_COMPARE_OP_GREATER": 1, "VK_COMPARE_OP_GREATER_OR_EQUAL": 1}


def clear_dir(value):
    """A depth clear to the far plane: 1.0 forward, 0.0 reversed."""
    if value is None:
        return None
    return {1.0: 0, 0.0: 1}.get(float(value))


# Clip-space depth range: [0, 1] unless VkPipelineViewportDepthClipControl
# CreateInfoEXT::negativeOneToOne is set.
def api_depth_range(depth_clip_negative_one_to_one=False):
    return 1 if depth_clip_negative_one_to_one else 0
