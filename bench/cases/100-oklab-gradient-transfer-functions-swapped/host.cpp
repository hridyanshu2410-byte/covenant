// Distilled from zed-industries/zed PR #57543 ("gpui_windows: Fix reversed
// linear/sRGB transfer functions in shader", fixes #57542, merged
// 2026-09-17). In the Windows (DirectX 11) renderer's shaders.hlsl,
// linear_to_srgb() raised to 2.2 and srgb_to_linear() raised to 1/2.2 -- the
// two transfer functions were swapped. They are only used by the Oklab
// gradient path: prepare_gradient_color() converts both stops with
// srgb_to_oklab() (which starts with srgb_to_linear), gradient_color() lerps
// in Oklab and converts back with oklab_to_srgb() (which ends with
// linear_to_srgb). Oklab gradients on Windows came out with the wrong
// midpoints and shifted endpoints compared with the Metal and WGSL
// renderers, which had the exponents the right way round. The fix swaps the
// exponents back.
// Port: HLSL/D3D11 -> Slang/Vulkan; quads come from a structured buffer and
// only the background fill is kept. The swapchain is B8G8R8A8_UNORM like
// gpui's DXGI swap chain, so the shader writes sRGB-encoded values.
#include "../../common/app.hpp"
using namespace bench;

struct Bounds { float origin[2]; float size[2]; };
struct Hsla { float h, s, l, a; };
struct LinearColorStop { Hsla color; float percentage; };
struct Background {
  uint32_t tag; uint32_t color_space; Hsla solid; float gradient_angle_or_pattern_height;
  LinearColorStop colors[2]; uint32_t pad;
};
struct Quad { uint32_t order; uint32_t border_style; Bounds bounds; Background background; };
static_assert(sizeof(Quad) == 4 * 2 + 16 + (8 + 16 + 4 + 40 + 4), "Quad layout");
struct GlobalParams { float global_viewport_size[2]; float pad[2]; };

static Hsla rgb_hex(uint32_t hex, float alpha) {   // gpui::rgb() -> Hsla
  float r = ((hex >> 16) & 0xff) / 255.f, g = ((hex >> 8) & 0xff) / 255.f, b = (hex & 0xff) / 255.f;
  float mx = std::max(r, std::max(g, b)), mn = std::min(r, std::min(g, b)), l = (mx + mn) / 2, d = mx - mn;
  float h = 0, s = 0;
  if (d > 0) {
    s = l > 0.5f ? d / (2 - mx - mn) : d / (mx + mn);
    if (mx == r) h = (g - b) / d + (g < b ? 6 : 0); else if (mx == g) h = (b - r) / d + 2; else h = (r - g) / d + 4;
    h /= 6;
  }
  return {h, s, l, alpha};
}

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;   // DXGI_FORMAT_B8G8R8A8_UNORM swap chain
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");

  std::vector<Quad> scene;
  auto add = [&](float x, float y, float w, float h, Background bg) {
    Quad q{}; q.order = (uint32_t)scene.size(); q.bounds = {{x, y}, {w, h}}; q.background = bg; scene.push_back(q);
  };
  Background panel{}; panel.tag = 0; panel.solid = rgb_hex(0x282c34, 1.0f);
  add(0, 0, (float)a.extent.width, (float)a.extent.height, panel);
  // linear_gradient(90., linear_color_stop(blue, 0.), linear_color_stop(red, 1.)).color_space(Oklab)
  Background grad{}; grad.tag = 1; grad.color_space = 1; grad.gradient_angle_or_pattern_height = 90.f;
  grad.colors[0] = {rgb_hex(0x0000ff, 1.0f), 0.0f};
  grad.colors[1] = {rgb_hex(0xff0000, 1.0f), 1.0f};
  add(40, 60, 560, 120, grad);
  // the same in sRGB interpolation for comparison
  Background grad_srgb = grad; grad_srgb.color_space = 0;
  add(40, 240, 560, 120, grad_srgb);

  Buffer quads = make_buffer(a, scene.size() * sizeof(Quad), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "quads");
  memcpy(quads.map, scene.data(), scene.size() * sizeof(Quad));
  Buffer globals = make_buffer(a, sizeof(GlobalParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "global_params");
  GlobalParams gp{{(float)a.extent.width, (float)a.extent.height}, {0, 0}};
  memcpy(globals.map, &gp, sizeof gp);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto set = alloc_set(a, setl, "quad_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, quads);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, globals);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "quad_vertex"; gd.fs_entry = "quad_fragment";
  gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline quadPipeline = make_graphics(a, gd, "quad_pipeline");

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, quadPipeline.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, quadPipeline.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 6, (uint32_t)scene.size(), 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
