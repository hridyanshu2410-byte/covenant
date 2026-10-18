// Distilled from bkaradzic/bgfx issue #3851 / PR #3852 ("D3D12: Fixed
// submitBlit not transitioning destination texture to COPY_DEST", merged
// 2026-07-26). RendererContextD3D12::submitBlit() took the destination
// texture as const: it transitioned the source to COPY_SOURCE (and back) but
// issued CopyTextureRegion into the destination in whatever state the
// destination was last left in, e.g. PIXEL_SHADER_RESOURCE after it was
// sampled, or UNORDERED_ACCESS after a compute write. The debug layer
// reports the state mismatch ("Before state ... does not match with the
// state ... COPY_DEST"). The fix makes the destination non-const,
// transitions it to COPY_DEST before the copy and restores its previous
// state afterwards, as the Vulkan backend already did.
// Port: D3D12 resource states -> image layouts (PIXEL_SHADER_RESOURCE ->
// SHADER_READ_ONLY_OPTIMAL, COPY_SOURCE/DEST -> TRANSFER_SRC/DST_OPTIMAL).
// The destination is a texture sampled every frame and updated by a blit
// from a second texture each frame (bgfx::blit in a view before the draw).
#include "../../common/app.hpp"
using namespace bench;

struct TextureD3D12 { Image img; VkImageLayout state; };

static void set_state(VkCommandBuffer c, TextureD3D12& t, VkImageLayout to, VkPipelineStageFlags2 srcStage,
                      VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
  if (t.state == to) return;
  image_barrier(c, t.img.img, t.state, to, srcStage, srcAccess, dstStage, dstAccess);
  t.state = to;
}

// RendererContextD3D12::submitBlit
#ifdef FIXED
static void submit_blit(VkCommandBuffer c, TextureD3D12& src, TextureD3D12& dst) {
#else
static void submit_blit(VkCommandBuffer c, TextureD3D12& src, const TextureD3D12& dst) {
#endif
  VkImageLayout srcState = src.state;
  set_state(c, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
#ifdef FIXED
  VkImageLayout dstState = dst.state;
  set_state(c, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
#endif
  VkImageCopy region{};
  region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.extent = {64, 64, 1};
  vkCmdCopyImage(c, src.img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 1, &region);                                                  // CopyTextureRegion
#ifdef FIXED
  set_state(c, dst, dstState, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
#endif
  set_state(c, src, srcState, VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  TextureD3D12 src{make_image(a, 64, 64, VK_FORMAT_R8G8B8A8_UNORM, usage, "blit_source"), VK_IMAGE_LAYOUT_UNDEFINED};
  TextureD3D12 dst{make_image(a, 64, 64, VK_FORMAT_R8G8B8A8_UNORM, usage, "blit_destination"), VK_IMAGE_LAYOUT_UNDEFINED};

  // texture creation: both start sampled (D3D12 resources created, then moved to PIXEL_SHADER_RESOURCE)
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  set_state(up, src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  set_state(up, dst, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "atlas_srv");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, dst.img.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline quad = make_graphics(a, gd, "textured_quad");

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    submit_blit(c, src, dst);                             // view 0: bgfx::blit(view, dst, ..., src)
    begin_rendering(a, c, {a.swap_views[idx]});           // view 1: draw sampling dst
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, quad.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, quad.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
