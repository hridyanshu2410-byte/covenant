// Distilled from SaschaWillems/Vulkan #667 (computeshader example): the
// storage image that the compute shader writes is created with
// VK_SHARING_MODE_EXCLUSIVE. It is moved to GENERAL on the graphics queue,
// written by a dispatch on the compute queue (a different family) and
// sampled by the fragment shader on the graphics queue, with no queue family
// ownership transfer anywhere. A semaphore orders the queues. The sample's
// later code (the fix): when the graphics and compute families differ,
// create the image VK_SHARING_MODE_CONCURRENT with both family indices.
#include "../../common/app.hpp"
using namespace bench;

struct ComputeParams { float time; float strength; float pad[2]; };

static Image create_storage_image(App& a, uint32_t w, uint32_t h, VkFormat fmt) {
  Image im{}; im.format = fmt; im.w = w; im.h = h; im.mips = 1;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {w, h, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  // Image will be sampled in the fragment shader and used as storage target in the compute shader
  ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
#ifdef FIXED
  // If compute and graphics queue family indices differ, create an image that can be shared between them
  uint32_t families[2] = {a.gfx_family, a.comp_family};
  if (a.gfx_family != a.comp_family) {
    ii.sharingMode = VK_SHARING_MODE_CONCURRENT;
    ii.queueFamilyIndexCount = 2; ii.pQueueFamilyIndices = families;
  }
#endif
  VK(vkCreateImage(a.dev, &ii, nullptr, &im.img));
  VkMemoryRequirements r; vkGetImageMemoryRequirements(a.dev, im.img, &r);
  VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mi.allocationSize = r.size; mi.memoryTypeIndex = mem_type(a, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK(vkAllocateMemory(a.dev, &mi, nullptr, &im.mem));
  VK(vkBindImageMemory(a.dev, im.img, im.mem, 0));
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = im.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VK(vkCreateImageView(a.dev, &vi, nullptr, &im.view));
  a.name(im.img, VK_OBJECT_TYPE_IMAGE, "storage_image");
  a.name(im.view, VK_OBJECT_TYPE_IMAGE_VIEW, "storage_image");
  return im;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);                       // graphics family 0, compute-only family 1
  Image storage = create_storage_image(a, 512, 512, VK_FORMAT_R8G8B8A8_UNORM);
  Buffer params = make_buffer(a, sizeof(ComputeParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "compute_params");
  ComputeParams p{0.0f, 1.0f, {0, 0}};
  memcpy(params.map, &p, sizeof p);

  auto sh = load_shader(a, "shaders.spv");
  auto csl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
                                 {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto cset = alloc_set(a, csl, "compute_set");
  write_buffer(a, cset, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, params);
  write_image(a, cset, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storage.view, VK_IMAGE_LAYOUT_GENERAL);
  Pipeline filter = make_compute(a, sh, "csMain", {csl}, "edgedetect");

  auto gsl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto gset = alloc_set(a, gsl, "post_compute_set");
  write_image(a, gset, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, storage.view, VK_IMAGE_LAYOUT_GENERAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {gsl}; gd.colors = {a.swap_format};
  Pipeline quad = make_graphics(a, gd, "textured_quad");

  // Transition image to the general layout, so we can use it as a storage image in the compute shader
  {
    VkCommandBuffer layout_cmd = alloc_cmd(a, a.gfx_pool);
    begin(layout_cmd);
    image_barrier(layout_cmd, storage.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    VK(vkEndCommandBuffer(layout_cmd));
    VkFence f = make_fence(a, false, "layout_fence");
    submit(a.gfx, layout_cmd, {}, {}, f);
    VK(vkWaitForFences(a.dev, 1, &f, VK_TRUE, UINT64_MAX));
  }

  VkCommandBuffer ccmd = alloc_cmd(a, a.comp_pool);
  VkFence cfence = make_fence(a, true, "compute_fence");
  VkSemaphore compute_done = make_semaphore(a, "compute_finished");
  VkSemaphore graphics_done = make_semaphore(a, "graphics_finished");
  const int FRAMES = frames_from_env(4);
  VkCommandBuffer gcmd = alloc_cmd(a, a.gfx_pool);
  VkFence gfence = make_fence(a, true, "graphics_fence");
  VkSemaphore acquired = make_semaphore(a, "present_complete");
  std::vector<VkSemaphore> render_complete;
  for (size_t i = 0; i < a.swap_images.size(); i++)
    render_complete.push_back(make_semaphore(a, "render_complete[" + std::to_string(i) + "]"));

  for (int f = 0; f < FRAMES; f++) {
    // compute: filter into the storage image
    VK(vkWaitForFences(a.dev, 1, &cfence, VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &cfence));
    begin(ccmd);
    vkCmdBindPipeline(ccmd, VK_PIPELINE_BIND_POINT_COMPUTE, filter.pipe);
    vkCmdBindDescriptorSets(ccmd, VK_PIPELINE_BIND_POINT_COMPUTE, filter.layout, 0, 1, &cset, 0, nullptr);
    vkCmdDispatch(ccmd, 512 / 16, 512 / 16, 1);
    VK(vkEndCommandBuffer(ccmd));
    std::vector<std::pair<VkSemaphore, VkPipelineStageFlags>> cw;
    if (f > 0) cw.push_back({graphics_done, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT});
    submit(a.comp, ccmd, cw, {compute_done}, cfence);

    // graphics: sample the filtered image
    VK(vkWaitForFences(a.dev, 1, &gfence, VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &gfence));
    uint32_t idx;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &idx));
    begin(gcmd);
    // make sure that compute shader writes are finished before sampling from the texture
    image_barrier(gcmd, storage.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    image_barrier(gcmd, a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, gcmd, {a.swap_views[idx]});
    vkCmdBindPipeline(gcmd, VK_PIPELINE_BIND_POINT_GRAPHICS, quad.pipe);
    vkCmdBindDescriptorSets(gcmd, VK_PIPELINE_BIND_POINT_GRAPHICS, quad.layout, 0, 1, &gset, 0, nullptr);
    vkCmdDraw(gcmd, 3, 1, 0, 0);
    vkCmdEndRendering(gcmd);
    image_barrier(gcmd, a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    VK(vkEndCommandBuffer(gcmd));
    submit(a.gfx, gcmd, {{compute_done, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT},
                         {acquired, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}},
           {graphics_done, render_complete[idx]}, gfence);
    present(a, a.gfx, idx, render_complete[idx]);
  }
  vkDeviceWaitIdle(a.dev);
  return 0;
}
