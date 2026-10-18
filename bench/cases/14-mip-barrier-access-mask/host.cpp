// Distilled from techrote/ShadeDoomVK #83: the mipmap generation loop's
// barrier after each blit used the wrong access mask, so the next level's
// blit could read data that was not yet visible.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  const uint32_t MIPS = 4;
  Image tex = make_image(a, 256, 256, VK_FORMAT_R8G8B8A8_SRGB,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         "albedo", MIPS);
  Buffer staging = make_buffer(a, 256 * 256 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "albedo_staging");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "albedo_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, tex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "textured");

  // one-time upload + mip generation
  VkCommandBuffer c = alloc_cmd(a, a.gfx_pool);
  begin(c);
  image_barrier(c, tex.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkBufferImageCopy copy{}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = {256, 256, 1};
  vkCmdCopyBufferToImage(c, staging.buf, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  int32_t w = 256, h = 256;
  for (uint32_t i = 1; i < MIPS; i++) {
    // level i-1 was just written (copy or blit); make it the blit source
#ifdef FIXED
    VkAccessFlags2 src_access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
#else
    VkAccessFlags2 src_access = VK_ACCESS_2_TRANSFER_READ_BIT;    // BUG: should be the write
#endif
    image_barrier(c, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, src_access, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_TRANSFER_READ_BIT, i - 1, 1);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
    blit.srcOffsets[1] = {w, h, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
    blit.dstOffsets[1] = {w > 1 ? w / 2 : 1, h > 1 ? h / 2 : 1, 1};
    vkCmdBlitImage(c, tex.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit, VK_FILTER_LINEAR);
    image_barrier(c, tex.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, i - 1, 1);
    w /= 2; h /= 2;
  }
  image_barrier(c, tex.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, MIPS - 1, 1);
  VK(vkEndCommandBuffer(c));
  VkFence up = make_fence(a, false, "upload_fence");
  submit(a.gfx, c, {}, {}, up);
  VK(vkWaitForFences(a.dev, 1, &up, VK_TRUE, UINT64_MAX));

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
