// Distilled from ZihanWG/VulkanEngine PR #28: glTF textures are uploaded on a
// dedicated transfer queue of another family and handed to graphics with a
// queue family ownership transfer (release on the upload queue, acquire on
// graphics, same layouts) joined by a semaphore. The semaphore was signaled
// with stageMask ALL_TRANSFER, but the release barrier's layout transition is
// not part of the transfer stage (its second scope is ignored for a release),
// so the transition was not ordered before the acquire's transition:
// WRITE_AFTER_WRITE per texture. The fix changes the signal to ALL_COMMANDS.
#include "../../common/app.hpp"
using namespace bench;

static void qfot_image_barrier(VkCommandBuffer c, VkImage img, uint32_t src_family, uint32_t dst_family,
                               VkImageLayout from, VkImageLayout to,
                               VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                               VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.srcStageMask = src_stage; b.srcAccessMask = src_access; b.dstStageMask = dst_stage; b.dstAccessMask = dst_access;
  b.oldLayout = from; b.newLayout = to; b.image = img;
  b.srcQueueFamilyIndex = src_family; b.dstQueueFamilyIndex = dst_family;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
  vkCmdPipelineBarrier2(c, &di);
}

static void submit2(VkQueue q, VkCommandBuffer c, VkSemaphore wait, VkPipelineStageFlags2 wait_stage,
                    VkSemaphore signal, VkPipelineStageFlags2 signal_stage, VkFence fence) {
  VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}; cbi.commandBuffer = c;
  VkSemaphoreSubmitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}; wi.semaphore = wait; wi.stageMask = wait_stage;
  VkSemaphoreSubmitInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}; si.semaphore = signal; si.stageMask = signal_stage;
  VkSubmitInfo2 s{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  s.waitSemaphoreInfoCount = wait ? 1 : 0; s.pWaitSemaphoreInfos = &wi;
  s.commandBufferInfoCount = 1; s.pCommandBufferInfos = &cbi;
  s.signalSemaphoreInfoCount = signal ? 1 : 0; s.pSignalSemaphoreInfos = &si;
  VK(vkQueueSubmit2(q, 1, &s, fence));
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  const uint32_t upload_family = a.comp_family, gfx_family = a.gfx_family;   // transfer-capable family 1
  Image tex = make_image(a, 256, 256, VK_FORMAT_R8G8B8A8_SRGB,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "gltf_base_color");
  Buffer staging = make_buffer(a, 256 * 256 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "texture_staging");
  memset(staging.map, 0x80, 256 * 256 * 4);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "material_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "gltf_material");

  // upload queue: copy + release
  VkSemaphore uploaded = make_semaphore(a, "texture_uploaded");
  VkCommandBuffer uc = alloc_cmd(a, a.comp_pool);
  begin(uc);
  image_barrier(uc, tex.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkBufferImageCopy copy{}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = {256, 256, 1};
  vkCmdCopyBufferToImage(uc, staging.buf, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  qfot_image_barrier(uc, tex.img, upload_family, gfx_family,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_NONE, 0);                      // release: second scope ignored
  VK(vkEndCommandBuffer(uc));
#ifdef FIXED
  VkPipelineStageFlags2 signal_scope = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
#else
  VkPipelineStageFlags2 signal_scope = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;   // BUG: misses the release transition
#endif
  submit2(a.comp, uc, VK_NULL_HANDLE, 0, uploaded, signal_scope, VK_NULL_HANDLE);

  // graphics queue: acquire, then the first frame samples the texture
  VkCommandBuffer ac = alloc_cmd(a, a.gfx_pool);
  begin(ac);
  qfot_image_barrier(ac, tex.img, upload_family, gfx_family,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, 0,                       // acquire: first scope ignored
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  VK(vkEndCommandBuffer(ac));
  VkFence acf = make_fence(a, false, "acquire_fence");
  submit2(a.gfx, ac, uploaded, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_NULL_HANDLE, 0, acf);
  VK(vkWaitForFences(a.dev, 1, &acf, VK_TRUE, UINT64_MAX));

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
