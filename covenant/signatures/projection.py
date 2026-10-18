"""Perspective-projection builders whose body the host analysis does not see
(a library call): which argument is the vertical field of view and in which
unit. In-tree builders need no entry: host_meaning recognises them by their
body (tan / cos / sin of half a parameter, in a function that builds a 4x4
matrix from at least four parameters), so this table is only the out-of-tree
backstop. Each row is taken from the library's own documentation or source:

* GLM 0.9.9 / 1.0, glm/ext/matrix_clip_space.hpp: perspective*, perspectiveFov*,
  infinitePerspective*: fovy in radians (GLM_FORCE_RADIANS only since 0.9.6;
  the body computes tan(fovy / 2)).
* DirectXMath, XMMatrixPerspectiveFovLH / RH (FovAngleY, AspectRatio, NearZ,
  FarZ): "Top-down field-of-view angle in radians".
* cglm, cam.h glm_perspective(fovy, aspect, nearZ, farZ, dest) and the
  _lh/_rh/_no/_zo variants: "field of view angle (radians)".
* gl-matrix mat4.perspective(out, fovy, aspect, near, far) and
  perspectiveNO / perspectiveZO: "Vertical field of view in radians".
* linmath.h mat4x4_perspective(m, y_fov, aspect, n, f): body tanf(y_fov / 2).
* raylib raymath.h MatrixPerspective(fovY, aspect, nearPlane, farPlane):
  radians (body tan(fovY * 0.5)).
* HandmadeMath 2.x HMM_Perspective_RH_NO / RH_ZO / LH_NO / LH_ZO(FOV, ...):
  radians (1.x took degrees; 2.0 changed it).
* glam (Rust) Mat4::perspective_rh / _lh / _rh_gl / perspective_infinite_* /
  perspective_infinite_reverse_*(fov_y_radians, ...): radians; also used by
  C++ ports that keep glam's names.
* nalgebra Perspective3::new(aspect, fovy, znear, zfar): radians.
* bgfx bx::mtxProj(result, fovy, aspect, near, far, homogeneousNdc): degrees.
* Godot Projection::create_perspective(p_fovy_degrees, ...): degrees.
"""
GLM_FOV = {"perspective", "perspectiveRH", "perspectiveLH", "perspectiveFov",
           "perspectiveFovRH", "perspectiveFovLH", "infinitePerspective",
           "infinitePerspectiveRH", "infinitePerspectiveLH"}

DOC = {
    "dxm": "DirectXMath XMMatrixPerspectiveFov*: FovAngleY in radians",
    "cglm": "cglm glm_perspective*: fovy in radians",
    "glmat": "gl-matrix mat4.perspective*: fovy in radians",
    "linmath": "linmath.h mat4x4_perspective: y_fov in radians",
    "raylib": "raymath MatrixPerspective: fovY in radians",
    "hmm": "HandmadeMath 2.x HMM_Perspective_*: FOV in radians",
    "glam": "glam Mat4::perspective_*: fov_y_radians",
    "nalgebra": "nalgebra Perspective3::new(aspect, fovy, ...): radians",
    "bx": "bx::mtxProj: fovy in degrees",
    "godot": "Godot Projection::create_perspective: p_fovy_degrees",
}

# unqualified (or namespace-qualified) name -> (argument index, unit, citation)
BY_NAME = {
    "XMMatrixPerspectiveFovLH": (0, "rad", DOC["dxm"]),
    "XMMatrixPerspectiveFovRH": (0, "rad", DOC["dxm"]),
    "glm_perspective": (0, "rad", DOC["cglm"]),
    "glm_perspective_lh_no": (0, "rad", DOC["cglm"]), "glm_perspective_lh_zo": (0, "rad", DOC["cglm"]),
    "glm_perspective_rh_no": (0, "rad", DOC["cglm"]), "glm_perspective_rh_zo": (0, "rad", DOC["cglm"]),
    "mat4x4_perspective": (1, "rad", DOC["linmath"]),
    "MatrixPerspective": (0, "rad", DOC["raylib"]),
    "HMM_Perspective_RH_NO": (0, "rad", DOC["hmm"]), "HMM_Perspective_RH_ZO": (0, "rad", DOC["hmm"]),
    "HMM_Perspective_LH_NO": (0, "rad", DOC["hmm"]), "HMM_Perspective_LH_ZO": (0, "rad", DOC["hmm"]),
    "perspective_rh": (0, "rad", DOC["glam"]), "perspective_lh": (0, "rad", DOC["glam"]),
    "perspective_rh_gl": (0, "rad", DOC["glam"]),
    "perspective_infinite_rh": (0, "rad", DOC["glam"]), "perspective_infinite_lh": (0, "rad", DOC["glam"]),
    "perspective_infinite_reverse_rh": (0, "rad", DOC["glam"]),
    "perspective_infinite_reverse_lh": (0, "rad", DOC["glam"]),
    "create_perspective": (0, "deg", DOC["godot"]),
}
# names that are only distinctive with their namespace
BY_QUALIFIED = {
    ("bx", "mtxProj"): (1, "deg", DOC["bx"]),
    ("nalgebra", "new_perspective"): (1, "rad", DOC["nalgebra"]),
}


def lookup(name, ns):
    """(argument index, 'rad' | 'deg', citation) of the field of view of a
    library projection call, or None."""
    got = BY_QUALIFIED.get((ns.split("::")[-1] if ns else "", name))
    if got is not None:
        return got
    return BY_NAME.get(name)
