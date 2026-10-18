// Distilled from NVIDIAGameWorks/Falcor #102: an app used two render contexts
// on two queues; the second one updated a texture every few frames that the
// main context then used, and the only synchronization was the main
// context's frame fence. The second queue's work and the main queue's use of
// the texture were unordered on the GPU ("resource still referenced by
// operations in flight on another queue"). The maintainers' fix: submit the
// second context before the main one and order the main queue after it; in
// Vulkan terms, the main submit waits on a semaphore the second submit signals.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  const int FRAMES = frames_from_env(6);
  // The texture is shared by both queues (CONCURRENT, like a D3D12 resource:
  // no ownership transfer involved; the bug is ordering only).
  Image tex{}; tex.format = VK_FORMAT_R8G8B8A8_UNORM; tex.w = tex.h = 256; tex.mips = 1;
  {
    uint32_t fams[2] = {a.gfx_family, a.comp_family};
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = tex.format; ii.extent = {256, 256, 1}; ii.mipLevels = 1;
    ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.sharingMode = a.gfx_family != a.comp_family ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    ii.queueFamilyIndexCount = a.gfx_family != a.comp_family ? 2 : 0; ii.pQueueFamilyIndices = fams;
    VK(vkCreateImage(a.dev, &ii, nullptr, &tex.img));
    VkMemoryRequirements r; vkGetImageMemoryRequirements(a.dev, tex.img, &r);
    VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mi.allocationSize = r.size; mi.memoryTypeIndex = mem_type(a, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(a.dev, &mi, nullptr, &tex.mem));
    VK(vkBindImageMemory(a.dev, tex.img, tex.mem, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = tex.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = tex.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK(vkCreateImageView(a.dev, &vi, nullptr, &tex.view));
    a.name(tex.img, VK_OBJECT_TYPE_IMAGE, "overlay_tex");
    a.name(tex.view, VK_OBJECT_TYPE_IMAGE_VIEW, "overlay_tex");
  }
  auto sh = load_shader(a, "shaders.spv");
  auto csl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto cset = alloc_set(a, csl, "overlay_write_set");
  write_image(a, cset, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, tex.view, VK_IMAGE_LAYOUT_GENERAL);
  Pipeline update = make_compute(a, sh, "updateMain", {csl}, "overlay_update");
  auto gsl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto gset = alloc_set(a, gsl, "overlay_read_set");
  write_image(a, gset, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex.view, VK_IMAGE_LAYOUT_GENERAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {gsl}; gd.colors = {a.swap_format};
  Pipeline draw = make_graphics(a, gd, "composite");

  VkCommandBuffer main_cmd = alloc_cmd(a, a.gfx_pool), second_cmd = alloc_cmd(a, a.comp_pool);
  VkSemaphore acquired = make_semaphore(a, "acquired");
  std::vector<VkSemaphore> done(a.swap_images.size());  // one per swapchain image
  for (size_t i = 0; i < done.size(); i++) done[i] = make_semaphore(a, "render_done[" + std::to_string(i) + "]");
  VkSemaphore second_done = make_semaphore(a, "second_context_done");
  VkFence frame_fence = make_fence(a, true, "frame_fence");
  for (int f = 0; f < FRAMES; f++) {
    VK(vkWaitForFences(a.dev, 1, &frame_fence, VK_TRUE, UINT64_MAX));   // the only CPU sync
    VK(vkResetFences(a.dev, 1, &frame_fence));
    uint32_t idx;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &idx));
    bool update_now = (f % 3 == 0);              // second context works every 3rd frame
    if (update_now) {
      begin(second_cmd);
      image_barrier(second_cmd, tex.img, f == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
      vkCmdBindPipeline(second_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, update.pipe);
      vkCmdBindDescriptorSets(second_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, update.layout, 0, 1, &cset, 0, nullptr);
      vkCmdDispatch(second_cmd, 32, 32, 1);
      VK(vkEndCommandBuffer(second_cmd));
#ifdef FIXED
      submit(a.comp, second_cmd, {}, {second_done}, VK_NULL_HANDLE);
#else
      submit(a.comp, second_cmd, {}, {}, VK_NULL_HANDLE);          // BUG: nothing orders the main queue after it
#endif
    }
    begin(main_cmd);
    image_barrier(main_cmd, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, main_cmd, {a.swap_views[idx]});
    vkCmdBindPipeline(main_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.pipe);
    vkCmdBindDescriptorSets(main_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.layout, 0, 1, &gset, 0, nullptr);
    vkCmdDraw(main_cmd, 3, 1, 0, 0);
    vkCmdEndRendering(main_cmd);
    image_barrier(main_cmd, a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    VK(vkEndCommandBuffer(main_cmd));
    std::vector<std::pair<VkSemaphore, VkPipelineStageFlags>> waits = {{acquired, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}};
#ifdef FIXED
    if (update_now) waits.push_back({second_done, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT});
#endif
    submit(a.gfx, main_cmd, waits, {done[idx]}, frame_fence);
    present(a, a.gfx, idx, done[idx]);
  }
  vkDeviceWaitIdle(a.dev);
  return 0;
}
