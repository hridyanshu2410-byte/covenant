// Distilled from flutter/flutter PR #181340: a host struct mirroring a
// uniform block treated vec3 as 12 bytes; the shader aligns vec3 to 16, so
// every field after the first vec3 was read from the wrong offset.
#include "../../common/app.hpp"
using namespace bench;

#ifdef FIXED
struct alignas(16) FrameInfo { float lightDir[3]; float intensity; float tint[3]; float time; };
#else
struct FrameInfo { float lightDir[3]; float tint[3]; float intensity; float time; };   // BUG: order/padding
#endif

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Buffer ubo = make_buffer(a, sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "frame_info");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "frame_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "pulse");
  FrameInfo info{};
  run_frames(a, frames_from_env(), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) { info.time = f * 0.016f; info.intensity = 1; memcpy(ubo.map, &info, sizeof info); });
}
