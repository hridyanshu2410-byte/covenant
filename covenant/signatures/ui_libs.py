"""Immediate-mode UI library vertex conventions (from each library's
documentation, never from a bug).

Dear ImGui (imgui.h, ImDrawData / ImDrawVert): vertex positions are in
display ("screen space") pixel coordinates; ImDrawData::DisplayPos is the
top-left of the viewport and y grows downwards. Texture coordinates follow the
font atlas rows (v = 0 is the first row). A backend must therefore map
increasing y to increasing framebuffer y (clip_y: +Y down at the input).
"""
# vertex struct name -> {position field: orientation of its y axis}
UI_VERTEX = {
    "ImDrawVert": {"pos": "y_down"},
}
