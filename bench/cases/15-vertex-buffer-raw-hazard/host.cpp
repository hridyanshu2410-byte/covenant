// Distilled from gfx-rs/wgpu #6344 (SYNC-HAZARD-READ-AFTER-WRITE on a vertex
// buffer): a compute pass updated particles in a buffer that the next draw
// consumed as vertices, and the barrier between them targeted shader reads
// instead of vertex-attribute reads.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Buffer parts = make_buffer(a, 1024 * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             false, "particles");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto set = alloc_set(a, setl, "sim_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, parts);
  auto sh = load_shader(a, "shaders.spv");
  Pipeline sim = make_compute(a, sh, "simMain", {setl}, "particle_sim");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.colors = {a.swap_format}; gd.vertex_stride = 16;
  gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32_SFLOAT, 12}};
  Pipeline draw = make_graphics(a, gd, "particle_draw");
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, sim.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, sim.layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(c, 16, 1, 1);
#ifdef FIXED
    buffer_barrier(c, parts.buf, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT);
#else
    buffer_barrier(c, parts.buf, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);   // BUG: wrong scope
#endif
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.pipe);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &parts.buf, &off);
    vkCmdDraw(c, 1024, 1, 0, 0);
    vkCmdEndRendering(c);
    // next frame's compute writes what this draw read
    buffer_barrier(c, parts.buf, VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  });
}
