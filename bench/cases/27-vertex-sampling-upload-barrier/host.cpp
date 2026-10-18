// Distilled from flutter/flutter #193371: Impeller's Vulkan backend uploads a
// texture (copyBufferToTexture / Texture.overwrite) and transitions it to
// SHADER_READ_ONLY with dst_stage = eFragmentShader only. Once the image is in
// SHADER_READ_ONLY, binding it emits no further barrier, so nothing orders a
// vertex shader read after the transfer write. Flutter GPU allows vertex-stage
// sampling (heightfields, texture-driven skinning). The fix the issue asks for:
// include the vertex shader stage in the barrier's destination scope.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Image height = make_image(a, 128, 128, VK_FORMAT_R8_UNORM,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "heightfield");
  Buffer staging = make_buffer(a, 128 * 128, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "heightfield_staging");
  memset(staging.map, 0x40, 128 * 128);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto set = alloc_set(a, setl, "heightfield_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, height.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "terrain");
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0) {   // BlitPassVK::OnCopyBufferToTextureCommand, same command buffer as the render pass
      image_barrier(c, height.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      VkBufferImageCopy copy{}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = {128, 128, 1};
      vkCmdCopyBufferToImage(c, staging.buf, height.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
#ifdef FIXED
      VkPipelineStageFlags2 dst = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
#else
      VkPipelineStageFlags2 dst = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;        // BUG: fragment only
#endif
      image_barrier(c, height.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, dst, VK_ACCESS_2_SHADER_READ_BIT);
    }
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
