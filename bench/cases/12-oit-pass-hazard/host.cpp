// Distilled from flycast PR #2507: order-independent transparency wrote a
// fragment list in one pass and resolved it in the next without a barrier
// between them (sync hazard flagged by validation).
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Buffer frags = make_buffer(a, 64 * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, "oit_fragments");
  Buffer counter = make_buffer(a, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, "oit_counter");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "oit_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frags);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, counter);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc ad; ad.vs = ad.fs = sh; ad.fs_entry = "appendMain"; ad.sets = {setl}; ad.colors = {};
  Pipeline append = make_graphics(a, ad, "oit_append");
  GfxDesc rd; rd.vs = rd.fs = sh; rd.fs_entry = "resolveMain"; rd.sets = {setl}; rd.colors = {a.swap_format};
  Pipeline resolve = make_graphics(a, rd, "oit_resolve");
  run_frames(a, frames_from_env(4), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    vkCmdFillBuffer(c, counter.buf, 0, 16, 0);
    buffer_barrier(c, counter.buf, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    begin_rendering(a, c, {});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, append.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, append.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
#ifdef FIXED
    buffer_barrier(c, frags.buf, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
#endif
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, resolve.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, resolve.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    // the next frame clears the counter after this frame's passes
    buffer_barrier(c, counter.buf, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    buffer_barrier(c, frags.buf, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  });
}
