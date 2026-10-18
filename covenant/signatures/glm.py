"""GLM 0.9.9 / 1.0 signatures (glm/ext/matrix_clip_space.hpp,
matrix_transform.hpp, trigonometric.hpp, matrix.hpp).

* Projections build OpenGL-convention clip space: +Y up (clip_y 0). Their
  clip depth range is [-1, 1] (depth.range 1) unless the _ZO variant is
  called or GLM_FORCE_DEPTH_ZERO_TO_ONE is defined before the first GLM
  include (GLM_CONFIG_CLIP_CONTROL); _NO variants are always [-1, 1].
* fovy and other angle parameters are radians (GLM_FORCE_RADIANS is the only
  mode since 0.9.6).
* lookAt maps world -> view; inverse swaps a matrix's from / to frames.
* radians(): degrees -> radians; degrees(): the inverse.
"""
FORCE_ZO = "GLM_FORCE_DEPTH_ZERO_TO_ONE"
FORCE_LH = "GLM_FORCE_LEFT_HANDED"

PROJECTIONS = {"perspective", "perspectiveRH", "perspectiveLH", "perspectiveFov",
               "perspectiveFovRH", "perspectiveFovLH", "frustum", "frustumRH", "frustumLH",
               "ortho", "orthoRH", "orthoLH", "infinitePerspective", "infinitePerspectiveRH",
               "infinitePerspectiveLH", "tweakedInfinitePerspective"}
ANGLE_PARAMS = {"perspective": (0,), "perspectiveRH": (0,), "perspectiveLH": (0,),
                "perspectiveFov": (0,), "perspectiveFovRH": (0,), "perspectiveFovLH": (0,),
                "infinitePerspective": (0,), "rotate": (1,)}
CLIP_Y = 0                                    # GLM clip space is +Y up


def depth_range(fn, zo_defined):
    """depth.range element of glm::<fn>'s clip z (0 = [0,1], 1 = [-1,1])."""
    base = fn.split("::")[-1]
    if base.endswith("_ZO"):
        return 0
    if base.endswith("_NO"):
        return 1
    return 0 if zo_defined else 1


def is_projection(fn):
    base = fn.split("::")[-1]
    for suf in ("_ZO", "_NO"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base in PROJECTIONS


ANGLE_CONVERT = {"radians": -1, "degrees": +1}    # angle element added
FRAME_FUNCS = {"lookAt": ("world", "view"), "lookAtRH": ("world", "view"),
               "lookAtLH": ("world", "view")}
INVERSE = {"inverse", "affineInverse", "inverseTranspose"}
