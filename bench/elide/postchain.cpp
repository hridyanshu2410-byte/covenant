// Workload 3: "postchain" -- two independent post-processing chains, interleaved.
//
// A bloom chain (A0 -> A1 -> ... -> A5, 2x2 box filter) and a luminance chain
// (B0 -> ... -> B5, max filter) are recorded interleaved, A1 B1 A2 B2 ...,
// with a full barrier after every dispatch; a composite reads both. Each
// barrier alone is redundant (the next one also orders the chain), but not
// all of them together: every other barrier must stay. This is the case that
// needs the joint (greedy, re-verified) search rather than per-barrier checks.
#include "common.hpp"
#include "shaders_spv.h"

using namespace el;

int main(int argc, char** argv) {
  Ctx c = init(argc, argv, "postchain");
  const uint32_t W = (uint32_t)(1024 * std::sqrt(c.args.scale)) / 32 * 32, H = (uint32_t)(576 * std::sqrt(c.args.scale)) / 32 * 32;
  const uint32_t ITERS = 16;
  const int L = 6;
  auto use = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  Image A[L], B[L];
  for (int k = 0; k < L; k++) {
    A[k] = make_image(c, W >> k, H >> k, VK_FORMAT_R16G16B16A16_SFLOAT, use, "bloom" + std::to_string(k));
    B[k] = make_image(c, W >> k, H >> k, VK_FORMAT_R16G16B16A16_SFLOAT, use, "lum" + std::to_string(k));
  }
  Image out = make_image(c, W, H, VK_FORMAT_R8G8B8A8_UNORM, use, "post_out");
  Buffer params = make_buffer(c, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, "params");

  auto si = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, ub = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  VkDescriptorSetLayout l_src = set_layout(c, {ub, si}, VK_SHADER_STAGE_COMPUTE_BIT);
  VkDescriptorSetLayout l_down = set_layout(c, {si, si}, VK_SHADER_STAGE_COMPUTE_BIT);
  VkDescriptorSetLayout l_comp = set_layout(c, {si, si, si, si}, VK_SHADER_STAGE_COMPUTE_BIT);
  VkPipelineLayout pl_src = pipe_layout(c, l_src, 8, VK_SHADER_STAGE_COMPUTE_BIT);
  VkPipelineLayout pl_down = pipe_layout(c, l_down, 8, VK_SHADER_STAGE_COMPUTE_BIT);
  VkPipelineLayout pl_comp = pipe_layout(c, l_comp, 0, 0);
  VkPipeline p_src = compute_pipeline(c, pl_src, src_comp, sizeof src_comp, "chain_source");
  VkPipeline p_down = compute_pipeline(c, pl_down, down_comp, sizeof down_comp, "chain_step");
  VkPipeline p_comp = compute_pipeline(c, pl_comp, composite_comp, sizeof composite_comp, "composite");

  auto G = VK_IMAGE_LAYOUT_GENERAL;
  VkDescriptorSet d_srcA = alloc_set(c, l_src, "srcA_set"), d_srcB = alloc_set(c, l_src, "srcB_set");
  write_set(c, d_srcA, {{ub, params.buf}, {si, VK_NULL_HANDLE, A[0].view, G}});
  write_set(c, d_srcB, {{ub, params.buf}, {si, VK_NULL_HANDLE, B[0].view, G}});
  VkDescriptorSet d_A[L], d_B[L];
  for (int k = 1; k < L; k++) {
    d_A[k] = alloc_set(c, l_down, "downA_set" + std::to_string(k));
    d_B[k] = alloc_set(c, l_down, "downB_set" + std::to_string(k));
    write_set(c, d_A[k], {{si, VK_NULL_HANDLE, A[k - 1].view, G}, {si, VK_NULL_HANDLE, A[k].view, G}});
    write_set(c, d_B[k], {{si, VK_NULL_HANDLE, B[k - 1].view, G}, {si, VK_NULL_HANDLE, B[k].view, G}});
  }
  VkDescriptorSet d_comp = alloc_set(c, l_comp, "composite_set");
  write_set(c, d_comp, {{si, VK_NULL_HANDLE, A[L - 1].view, G}, {si, VK_NULL_HANDLE, B[L - 1].view, G},
                        {si, VK_NULL_HANDLE, A[0].view, G}, {si, VK_NULL_HANDLE, out.view, G}});

  auto dispatch_img = [&](VkCommandBuffer cb, const Image& dst) { vkCmdDispatch(cb, (dst.w + 7) / 8, (dst.h + 7) / 8, 1); };
  run_frames(c, [&](VkCommandBuffer cb, int f) {
    std::vector<VkImageMemoryBarrier2> tr;
    for (int k = 0; k < L; k++) {
      tr.push_back(image_barrier(A[k].img, VK_IMAGE_LAYOUT_UNDEFINED, G));
      tr.push_back(image_barrier(B[k].img, VK_IMAGE_LAYOUT_UNDEFINED, G));
    }
    tr.push_back(image_barrier(out.img, VK_IMAGE_LAYOUT_UNDEFINED, G));
    image_barriers(cb, tr);
    float p[4] = {0.37f * (float)f, 0.11f * (float)f, 0, 0};
    vkCmdUpdateBuffer(cb, params.buf, 0, 16, p);
    full_barrier_v1(cb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_src);
    for (uint32_t m = 0; m < 2; m++) {
      uint32_t pc[2] = {m, ITERS};
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_src, 0, 1, m ? &d_srcB : &d_srcA, 0, nullptr);
      vkCmdPushConstants(cb, pl_src, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
      dispatch_img(cb, A[0]);
      full_barrier(cb);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_down);
    for (int k = 1; k < L; k++) {
      for (uint32_t m = 0; m < 2; m++) {
        uint32_t pc[2] = {m, ITERS};
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_down, 0, 1, m ? &d_B[k] : &d_A[k], 0, nullptr);
        vkCmdPushConstants(cb, pl_down, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
        dispatch_img(cb, A[k]);
        full_barrier(cb);
      }
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_comp);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl_comp, 0, 1, &d_comp, 0, nullptr);
    dispatch_img(cb, out);
    full_barrier(cb);
  });
  std::vector<Readback> outs = {{"image", &out, nullptr, G}};
  for (int k = 0; k < L; k++) {                            // every level of both chains
    outs.push_back({"bloom" + std::to_string(k), &A[k], nullptr, G, 8});
    outs.push_back({"lum" + std::to_string(k), &B[k], nullptr, G, 8});
  }
  readback_and_hash(c, outs);
  finish(c);
  return 0;
}
