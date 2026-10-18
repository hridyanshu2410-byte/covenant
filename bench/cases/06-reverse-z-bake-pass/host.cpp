// Distilled from godotengine/godot #90627: after the renderer switched to
// reverse-Z, the LightmapGI bake pass kept the old depth test (LESS, clear
// to 1.0), so its depth test rejected the wrong surfaces.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Image depth = make_image(a, 640, 480, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "bake_depth");
  Image bake = make_image(a, 640, 480, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, "bake_target");
  Buffer cam = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto set = alloc_set(a, setl, "camera_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cam);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {VK_FORMAT_R16G16B16A16_SFLOAT};
  gd.depth = VK_FORMAT_D32_SFLOAT;
#ifdef FIXED
  gd.depth_compare = VK_COMPARE_OP_GREATER; float clear = 0.0f;
#else
  gd.depth_compare = VK_COMPARE_OP_LESS; float clear = 1.0f;     // BUG: forward-Z depth test
#endif
  Pipeline pipe = make_graphics(a, gd, "lightmap_bake");
  VkCommandBuffer c = alloc_cmd(a, a.gfx_pool);
  begin(c);
  image_barrier(c, bake.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
  begin_rendering(a, c, {bake.view}, depth.view, clear);
  vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
  vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
  vkCmdDraw(c, 3, 1, 0, 0);
  vkCmdEndRendering(c);
  VK(vkEndCommandBuffer(c));
  VkFence fn = make_fence(a, false, "bake_fence");
  submit(a.gfx, c, {}, {}, fn);
  VK(vkWaitForFences(a.dev, 1, &fn, VK_TRUE, UINT64_MAX));
}
