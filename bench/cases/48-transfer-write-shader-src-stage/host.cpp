// Distilled from o3de/o3de #11368 (open): with validation enabled, the
// Vulkan RHI records image barriers whose srcAccessMask is
// VK_ACCESS_TRANSFER_WRITE_BIT while srcStageMask holds only the shader
// stages (vertex, tessellation, geometry, fragment, compute), and in a second
// occurrence only VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
// (VUID-vkCmdPipelineBarrier-srcAccessMask-02815: the access is not
// supported by the stage mask). The write being made available is a
// transfer write, but the barrier's first scope does not contain the
// transfer stage, so it is not ordered after the copy. The issue does not
// name the resources; here they are a streamed texture filled by
// vkCmdCopyBufferToImage and a copy target filled by vkCmdCopyImage, both
// sampled afterwards. No upstream fix; the fixed variant uses
// VK_PIPELINE_STAGE_TRANSFER_BIT as the source stage, as the VUID requires.
#include "../../common/app.hpp"
using namespace bench;

static void barrier(VkCommandBuffer c, VkImage img, VkImageLayout from, VkImageLayout to,
                    VkPipelineStageFlags src_stage, VkAccessFlags src_access,
                    VkPipelineStageFlags dst_stage, VkAccessFlags dst_access) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcAccessMask = src_access; b.dstAccessMask = dst_access; b.oldLayout = from; b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(c, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static const VkPipelineStageFlags kShaderStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
    VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT |
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t W = 256, H = 256;
  Image streamed = make_image(a, W, H, VK_FORMAT_R8G8B8A8_SRGB,
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "streamed_texture");
  Image source = make_image(a, W, H, VK_FORMAT_R8G8B8A8_SRGB,
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "copy_source");
  Image copied = make_image(a, W, H, VK_FORMAT_R8G8B8A8_SRGB,
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "copy_target");
  Buffer staging = make_buffer(a, W * H * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "staging");
  memset(staging.map, 0x80, W * H * 4);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "material_srg");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, streamed.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, copied.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vsFullscreen"; gd.fs_entry = "fsMain"; gd.sets = {setl};
  gd.colors = {a.swap_format};
  Pipeline p = make_graphics(a, gd, "forward_pass");

#ifdef FIXED
  const VkPipelineStageFlags upload_src = VK_PIPELINE_STAGE_TRANSFER_BIT, copy_src = VK_PIPELINE_STAGE_TRANSFER_BIT;
#else
  const VkPipelineStageFlags upload_src = kShaderStages, copy_src = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
#endif

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0) {
      // streaming upload
      barrier(c, streamed.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
      VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {W, H, 1};
      vkCmdCopyBufferToImage(c, staging.buf, streamed.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
      barrier(c, streamed.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              upload_src, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
      // copy source filled from the same staging data
      barrier(c, source.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
      vkCmdCopyBufferToImage(c, staging.buf, source.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
      barrier(c, source.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_ACCESS_TRANSFER_READ_BIT);
    }
    // copy pass, every frame
    barrier(c, copied.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy ic{}; ic.srcSubresource = ic.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    ic.extent = {W, H, 1};
    vkCmdCopyImage(c, source.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, copied.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &ic);
    barrier(c, copied.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            copy_src, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
