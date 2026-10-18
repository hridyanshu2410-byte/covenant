// Distilled from maplibre/maplibre-native PR #4642 ("WebGPU: fix location
// indicator UBO binding", merged 2026-09-17). The WebGPU backend binds the
// shared uniform buffers in group 0: GlobalPaintParamsUBO at
// idGlobalPaintParamsUBO = 0 and the drawable's UBO at
// idLocationIndicatorDrawableUBO = 4 (bindings 1 and 2 carry the texture and
// sampler of the textured variant). The location_indicator WGSL declared
// @group(0) @binding(0) var<uniform> ubo: LocationIndicatorUBO, so it read
// the 48-byte global paint params as its 80-byte matrix + colour: the
// user-location puck was drawn with a garbage transform and colour (or not
// at all). The fix declares the UBO at @binding(4) in both the plain and
// the textured shader.
// Port: WebGPU bind group -> one descriptor set; untextured variant only.
#include "../../common/app.hpp"
using namespace bench;

// mbgl/shaders/webgpu/shader_group.hpp / layer_ubo.hpp
constexpr uint32_t idGlobalPaintParamsUBO = 0;
constexpr uint32_t idLocationIndicatorDrawableUBO = 4;

struct GlobalPaintParamsUBO {
  float pattern_atlas_texsize[2];
  float units_to_pixels[2];
  float world_size[2];
  float camera_to_center_distance;
  float symbol_fade_change;
  float aspect_ratio;
  float pixel_ratio;
  float map_zoom;
  float pad1;
};
static_assert(sizeof(GlobalPaintParamsUBO) == 48, "");

struct LocationIndicatorDrawableUBO { float matrix[16]; float color[4]; };
static_assert(sizeof(LocationIndicatorDrawableUBO) == 80, "");

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");

  Buffer global = make_buffer(a, sizeof(GlobalPaintParamsUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true,
                              "global_paint_params_ubo");
  GlobalPaintParamsUBO gp{{1024, 1024}, {320, -240}, {8192, 8192}, 1.5f, 1.0f, 640.0f / 480.0f, 2.0f, 14.2f, 0};
  memcpy(global.map, &gp, sizeof gp);
  Buffer drawable = make_buffer(a, sizeof(LocationIndicatorDrawableUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true,
                                "location_indicator_drawable_ubo");
  LocationIndicatorDrawableUBO li{};
  li.matrix[0] = 0.05f; li.matrix[5] = 0.05f * 640.0f / 480.0f; li.matrix[10] = 1; li.matrix[15] = 1;
  li.matrix[12] = 0.1f; li.matrix[13] = -0.2f;                                    // puck position
  li.color[0] = 0.12f; li.color[1] = 0.45f; li.color[2] = 0.95f; li.color[3] = 1;
  memcpy(drawable.map, &li, sizeof li);

  Buffer vb = make_buffer(a, 6 * 2 * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "puck_quad");
  float q[12] = {-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};
  memcpy(vb.map, q, sizeof q);

  const VkShaderStageFlags vf = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  auto setl = make_set_layout(a, {{idGlobalPaintParamsUBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vf},
                                  {idLocationIndicatorDrawableUBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vf}});
  auto set = alloc_set(a, setl, "uniform_bind_group");
  write_buffer(a, set, idGlobalPaintParamsUBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, global);
  write_buffer(a, set, idLocationIndicatorDrawableUBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, drawable);

  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  gd.vertex_stride = 8; gd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}};
  Pipeline p = make_graphics(a, gd, "location_indicator");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &zero);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
