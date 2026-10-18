// Workload 1: "particles" -- independent compute dispatches separated by full barriers.
//
// Eight independent particle systems are integrated by eight dispatches, each
// writing only its own buffer. An over-cautious engine puts an ALL_COMMANDS
// memory barrier after every dispatch (and after the histogram clear, the
// binning pass and the shading pass). Only the dependencies
// clear+integrate -> bin -> shade are real.
#include "common.hpp"
#include "shaders_spv.h"

using namespace el;

int main(int argc, char** argv) {
  Ctx c = init(argc, argv, "particles");
  const uint32_t N = (uint32_t)(65536 * c.args.scale) / 64 * 64 + 64;   // particles per system
  const uint32_t ITERS = 64;
  const uint32_t W = (uint32_t)(512 * std::sqrt(c.args.scale)), H = W;
  const int S = 8;

  Buffer sys[S];
  for (int s = 0; s < S; s++)
    sys[s] = make_buffer(c, (VkDeviceSize)N * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false,
                         "particles" + std::to_string(s));
  Buffer hist = make_buffer(c, 64 * 64 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, "histogram");
  Image out = make_image(c, W, H, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "particles_out");

  auto sb = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  VkDescriptorSetLayout l_int = set_layout(c, {sb}, VK_SHADER_STAGE_COMPUTE_BIT);
  VkDescriptorSetLayout l_bin = set_layout(c, {sb, sb, sb, sb, sb, sb, sb, sb, sb}, VK_SHADER_STAGE_COMPUTE_BIT);
  VkDescriptorSetLayout l_shade = set_layout(c, {sb, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}, VK_SHADER_STAGE_COMPUTE_BIT);
  VkPipelineLayout pl_int = pipe_layout(c, l_int, 16, VK_SHADER_STAGE_COMPUTE_BIT);
  VkPipelineLayout pl_bin = pipe_layout(c, l_bin, 4, VK_SHADER_STAGE_COMPUTE_BIT);
  VkPipelineLayout pl_shade = pipe_layout(c, l_shade, 0, 0);
  VkPipeline p_int = compute_pipeline(c, pl_int, integrate_comp, sizeof integrate_comp, "integrate");
  VkPipeline p_bin = compute_pipeline(c, pl_bin, bin_comp, sizeof bin_comp, "bin");
  VkPipeline p_shade = compute_pipeline(c, pl_shade, shade_comp, sizeof shade_comp, "shade");

  VkDescriptorSet d_int[S];
  for (int s = 0; s < S; s++) {
    d_int[s] = alloc_set(c, l_int, "integrate_set" + std::to_string(s));
    write_set(c, d_int[s], {{sb, sys[s].buf}});
  }
  VkDescriptorSet d_bin = alloc_set(c, l_bin, "bin_set");
  std::vector<Desc> bd;
  for (int s = 0; s < S; s++) bd.push_back({sb, sys[s].buf});
  bd.push_back({sb, hist.buf});
  write_set(c, d_bin, bd);
  VkDescriptorSet d_shade = alloc_set(c, l_shade, "shade_set");
  write_set(c, d_shade, {{sb, hist.buf}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE, out.view, VK_IMAGE_LAYOUT_GENERAL}});

  run_frames(c, [&](VkCommandBuffer cb, int f) {
    image_barriers(cb, {image_barrier(out.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL)});
    vkCmdFillBuffer(cb, hist.buf, 0, VK_WHOLE_SIZE, 0);
    full_barrier_v1(cb);                                   // sync1 spelling
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_int);
    for (int s = 0; s < S; s++) {
      uint32_t pc[4] = {N, (uint32_t)f, (uint32_t)s, ITERS};
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_int, 0, 1, &d_int[s], 0, nullptr);
      vkCmdPushConstants(cb, pl_int, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
      vkCmdDispatch(cb, N / 64, 1, 1);
      full_barrier(cb);                                    // "after every dispatch"
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_bin);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_bin, 0, 1, &d_bin, 0, nullptr);
    vkCmdPushConstants(cb, pl_bin, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &N);
    vkCmdDispatch(cb, N / 64, 1, 1);
    full_barrier(cb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_shade);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_shade, 0, 1, &d_shade, 0, nullptr);
    vkCmdDispatch(cb, (W + 7) / 8, (H + 7) / 8, 1);
    full_barrier(cb);                                      // "end of frame"
  });
  std::vector<Readback> outs = {{"image", &out, nullptr, VK_IMAGE_LAYOUT_GENERAL}, {"histogram", nullptr, &hist}};
  for (int s = 0; s < S; s++) outs.push_back({"particles" + std::to_string(s), nullptr, &sys[s]});
  readback_and_hash(c, outs);                              // every output of the frame
  finish(c);
  return 0;
}
