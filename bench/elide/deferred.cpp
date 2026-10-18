// Workload 2: "deferred" -- a G-buffer renderer with a barrier after every pass.
//
// Three G-buffer passes draw to three different targets, then a lighting pass
// samples them into an HDR target and a tonemap pass writes the LDR output.
// Like many engines, it emits a full ALL_COMMANDS memory barrier after every
// render pass, and does its layout transitions with ALL_COMMANDS masks.
// The only real dependencies: G-buffers -> lighting -> tonemap.
#include "common.hpp"
#include "shaders_spv.h"

using namespace el;

static void pass(VkCommandBuffer cb, Image& target, VkPipeline p, VkPipelineLayout pl, VkDescriptorSet set,
                 const void* pc, uint32_t pcb) {
  VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  ca.imageView = target.view; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  ca.clearValue.color = {{0, 0, 0, 1}};
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {target.w, target.h}}; ri.layerCount = 1;
  ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
  vkCmdBeginRendering(cb, &ri);
  VkViewport vp{0, 0, (float)target.w, (float)target.h, 0, 1};
  VkRect2D sc{{0, 0}, {target.w, target.h}};
  vkCmdSetViewport(cb, 0, 1, &vp);
  vkCmdSetScissor(cb, 0, 1, &sc);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
  if (set) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 0, nullptr);
  if (pcb) vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, pcb, pc);
  vkCmdDraw(cb, 3, 1, 0, 0);
  vkCmdEndRendering(cb);
}

int main(int argc, char** argv) {
  Ctx c = init(argc, argv, "deferred");
  const uint32_t W = (uint32_t)(640 * std::sqrt(c.args.scale)), H = (uint32_t)(360 * std::sqrt(c.args.scale));
  const uint32_t ITERS = 48;
  auto att = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  Image g[3] = {make_image(c, W, H, VK_FORMAT_R8G8B8A8_UNORM, att, "gbuf_albedo"),
                make_image(c, W, H, VK_FORMAT_R8G8B8A8_UNORM, att, "gbuf_normal"),
                make_image(c, W, H, VK_FORMAT_R8G8B8A8_UNORM, att, "gbuf_material")};
  Image hdr = make_image(c, W, H, VK_FORMAT_R16G16B16A16_SFLOAT, att, "hdr");
  Image ldr = make_image(c, W, H, VK_FORMAT_R8G8B8A8_UNORM, att, "ldr");
  VkSampler smp = nearest_sampler(c);

  auto cis = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  VkDescriptorSetLayout l_light = set_layout(c, {cis, cis, cis}, VK_SHADER_STAGE_FRAGMENT_BIT);
  VkDescriptorSetLayout l_tone = set_layout(c, {cis}, VK_SHADER_STAGE_FRAGMENT_BIT);
  VkPipelineLayout pl_g = pipe_layout(c, VK_NULL_HANDLE, 12, VK_SHADER_STAGE_FRAGMENT_BIT);
  VkPipelineLayout pl_light = pipe_layout(c, l_light, 0, 0);
  VkPipelineLayout pl_tone = pipe_layout(c, l_tone, 0, 0);
  VkPipeline p_g = graphics_pipeline(c, pl_g, fullscreen_vert, sizeof fullscreen_vert, gbuf_frag, sizeof gbuf_frag,
                                     VK_FORMAT_R8G8B8A8_UNORM, "gbuffer");
  VkPipeline p_light = graphics_pipeline(c, pl_light, fullscreen_vert, sizeof fullscreen_vert, light_frag, sizeof light_frag,
                                         VK_FORMAT_R16G16B16A16_SFLOAT, "lighting");
  VkPipeline p_tone = graphics_pipeline(c, pl_tone, fullscreen_vert, sizeof fullscreen_vert, tonemap_frag, sizeof tonemap_frag,
                                        VK_FORMAT_R8G8B8A8_UNORM, "tonemap");
  VkDescriptorSet d_light = alloc_set(c, l_light, "light_set");
  write_set(c, d_light, {{cis, VK_NULL_HANDLE, g[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, smp},
                         {cis, VK_NULL_HANDLE, g[1].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, smp},
                         {cis, VK_NULL_HANDLE, g[2].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, smp}});
  VkDescriptorSet d_tone = alloc_set(c, l_tone, "tonemap_set");
  write_set(c, d_tone, {{cis, VK_NULL_HANDLE, hdr.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, smp}});

  const auto U = VK_IMAGE_LAYOUT_UNDEFINED, CA = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             SR = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  run_frames(c, [&](VkCommandBuffer cb, int f) {
    image_barriers(cb, {image_barrier(g[0].img, U, CA), image_barrier(g[1].img, U, CA), image_barrier(g[2].img, U, CA),
                        image_barrier(hdr.img, U, CA), image_barrier(ldr.img, U, CA)});
    for (uint32_t i = 0; i < 3; i++) {
      uint32_t pc[3] = {i, ITERS, (uint32_t)f};
      pass(cb, g[i], p_g, pl_g, VK_NULL_HANDLE, pc, 12);
      full_barrier(cb);                                    // "after every pass"
    }
    image_barriers(cb, {image_barrier(g[0].img, CA, SR), image_barrier(g[1].img, CA, SR), image_barrier(g[2].img, CA, SR)});
    pass(cb, hdr, p_light, pl_light, d_light, nullptr, 0);
    full_barrier(cb);
    image_barriers(cb, {image_barrier(hdr.img, CA, SR)});
    pass(cb, ldr, p_tone, pl_tone, d_tone, nullptr, 0);
    full_barrier(cb);
  });
  readback_and_hash(c, {{"ldr", &ldr, nullptr, CA}, {"hdr", &hdr, nullptr, SR, 8}, {"albedo", &g[0], nullptr, SR},
                        {"normal", &g[1], nullptr, SR}, {"material", &g[2], nullptr, SR}});
  finish(c);
  return 0;
}
