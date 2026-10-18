// Distilled from ocornut/imgui #8176 (partially fixed by PR #8600): with the
// docking branch, ImGui_ImplVulkan_RenderWindow() renders a secondary
// platform window (a window dragged outside the main viewport) with dynamic
// rendering. Before vkCmdBeginRendering it moved the acquired swapchain image
// UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL with
// vkCmdPipelineBarrier(fd->CommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
// VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, ...), while the submit waits
// on the image-acquired semaphore only at COLOR_ATTACHMENT_OUTPUT. The
// barrier's first scope (TOP_OF_PIPE) does not chain off that wait, so the
// layout transition can run before the presentation engine has finished
// reading the image (SYNC-HAZARD-WRITE-AFTER-READ, prior usage
// SYNC_PRESENT_ENGINE_SYNCVAL_PRESENT_ACQUIRE_READ_SYNCVAL). PR #8600 changes
// the source stage to COLOR_ATTACHMENT_OUTPUT (| NONE), transitions the
// backbuffers UNDEFINED -> PRESENT_SRC_KHR once when the window's swapchain is
// created (srcStage BOTTOM_OF_PIPE), and uses PRESENT_SRC_KHR as oldLayout
// per frame.
#include "../../common/app.hpp"
using namespace bench;

static void barrier1(VkCommandBuffer c, VkImage img, VkImageLayout from, VkImageLayout to,
                     VkPipelineStageFlags src, VkPipelineStageFlags dst, VkAccessFlags src_access, VkAccessFlags dst_access) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcAccessMask = src_access; b.dstAccessMask = dst_access; b.oldLayout = from; b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(c, src, dst, 0, 0, nullptr, 0, nullptr, 1, &b);
}

struct ImDrawVert { float pos[2]; float uv[2]; uint32_t col; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);      // a.swapchain: the secondary viewport's ImGui_ImplVulkanH_Window
  auto sh = load_shader(a, "shaders.spv");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format}; gd.push_size = 16;
  gd.vertex_stride = sizeof(ImDrawVert);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32_SFLOAT, 8}, {2, 0, VK_FORMAT_R8G8B8A8_UNORM, 16}};
  Pipeline imgui = make_graphics(a, gd, "imgui_pipeline");

  // font atlas, uploaded once
  Image font = make_image(a, 64, 64, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "font_atlas");
  Buffer staging = make_buffer(a, 64 * 64 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "font_upload");
  memset(staging.map, 0xff, 64 * 64 * 4);
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  image_barrier(up, font.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; region.imageExtent = {64, 64, 1};
  vkCmdCopyBufferToImage(up, staging.buf, font.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  image_barrier(up, font.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
#ifdef FIXED
  // ImGui_ImplVulkanH_CreateWindowSwapChain(): backbuffers start in PRESENT_SRC_KHR
  for (VkImage img : a.swap_images)
    barrier1(up, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0);
#endif
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));
  auto font_set = alloc_set(a, setl, "font_descriptor_set");
  write_image(a, font_set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, font.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  Buffer vb = make_buffer(a, 6 * sizeof(ImDrawVert), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "viewport_vertex_buffer");
  ImDrawVert quad[6] = {{{20, 20}, {0, 0}, 0xff3a2a1au}, {{300, 20}, {1, 0}, 0xff3a2a1au}, {{20, 200}, {0, 1}, 0xff3a2a1au},
                        {{300, 20}, {1, 0}, 0xff3a2a1au}, {{300, 200}, {1, 1}, 0xff3a2a1au}, {{20, 200}, {0, 1}, 0xff3a2a1au}};
  memcpy(vb.map, quad, sizeof quad);

  // ImGui_ImplVulkanH_Frame per backbuffer, ImGui_ImplVulkanH_FrameSemaphores per SemaphoreCount (= ImageCount + 1)
  uint32_t image_count = (uint32_t)a.swap_images.size(), semaphore_count = image_count + 1;
  std::vector<VkCommandBuffer> frame_cmd(image_count);
  std::vector<VkFence> frame_fence(image_count);
  std::vector<VkSemaphore> image_acquired(semaphore_count), render_complete(semaphore_count);
  for (uint32_t i = 0; i < image_count; i++) {
    frame_cmd[i] = alloc_cmd(a, a.gfx_pool);
    frame_fence[i] = make_fence(a, true, "Frames[" + std::to_string(i) + "].Fence");
  }
  for (uint32_t i = 0; i < semaphore_count; i++) {
    image_acquired[i] = make_semaphore(a, "FrameSemaphores[" + std::to_string(i) + "].ImageAcquiredSemaphore");
    render_complete[i] = make_semaphore(a, "FrameSemaphores[" + std::to_string(i) + "].RenderCompleteSemaphore");
  }
  uint32_t semaphore_index = 0;

  int frames = frames_from_env(6);
  for (int f = 0; f < frames; f++) {
    // ImGui_ImplVulkan_RenderWindow()
    VkSemaphore acquired = image_acquired[semaphore_index], complete = render_complete[semaphore_index];
    uint32_t frame_index;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &frame_index));
    VK(vkWaitForFences(a.dev, 1, &frame_fence[frame_index], VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &frame_fence[frame_index]));
    VkCommandBuffer c = frame_cmd[frame_index];
    begin(c);
#ifdef FIXED
    barrier1(c, a.swap_images[frame_index], VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
             0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
#else
    barrier1(c, a.swap_images[frame_index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
             0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
#endif
    begin_rendering(a, c, {a.swap_views[frame_index]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, imgui.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, imgui.layout, 0, 1, &font_set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    float st[4] = {2.0f / a.extent.width, 2.0f / a.extent.height, -1.0f, -1.0f};
    vkCmdPushConstants(c, imgui.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, 16, st);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
    barrier1(c, a.swap_images[frame_index], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0);
    VK(vkEndCommandBuffer(c));
    submit(a.gfx, c, {{acquired, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {complete}, frame_fence[frame_index]);
    // ImGui_ImplVulkan_SwapBuffers()
    present(a, a.gfx, frame_index, complete);
    semaphore_index = (semaphore_index + 1) % semaphore_count;
  }
  vkDeviceWaitIdle(a.dev);
}
