// Distilled from shadps4-emu/shadPS4 PR #5193 ("video_core: Fix sparse
// backing offsets, barriers, and memory budgeting", merged 2026-09-30), the
// image-barrier part. The texture cache tracks each image's last layout,
// access mask and stage, and Image::GetBarriers(dst_layout, dst_mask,
// dst_stage) emits a barrier only when the requested layout or access mask
// differs from the tracked one. Two consecutive uses with the same state,
// e.g. two compute dispatches that both bind the image as a storage image
// (GENERAL, SHADER_READ | SHADER_WRITE), got no barrier at all, although the
// first one wrote the image: read-after-write and write-after-write between
// the dispatches. The fix also emits the barrier when the tracked access
// mask contains a write (TRANSFER_WRITE, SHADER_WRITE or MEMORY_WRITE), for
// both the per-subresource and the full-resource path; the full-resource
// path is reproduced here.
#include "../../common/app.hpp"
using namespace bench;

struct State { VkImageLayout layout; VkAccessFlags2 access_mask; VkPipelineStageFlags2 pl_stage; };
struct TrackedImage { Image image; State last_state; };

// Image::GetBarriers + Transit (full resource transition)
static void transit(VkCommandBuffer c, TrackedImage& im, VkImageLayout dst_layout, VkAccessFlags2 dst_mask,
                    VkPipelineStageFlags2 dst_stage) {
  const State& last = im.last_state;
#ifdef FIXED
  const VkAccessFlags2 write_flags = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  const bool is_write = (last.access_mask & write_flags) != 0;
  if (last.layout == dst_layout && last.access_mask == dst_mask && !is_write) return;
#else
  if (last.layout == dst_layout && last.access_mask == dst_mask) return;
#endif
  image_barrier(c, im.image.img, last.layout, dst_layout, last.pl_stage, last.access_mask, dst_stage, dst_mask);
  im.last_state = {dst_layout, dst_mask, dst_stage};
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  TrackedImage rt{make_image(a, 64, 64, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             "guest_storage_image"),
                  {VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT}};
  auto csl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto cset = alloc_set(a, csl, "cs_storage_image");
  write_image(a, cset, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, rt.image.view, VK_IMAGE_LAYOUT_GENERAL);
  auto fsl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto fset = alloc_set(a, fsl, "ps_texture");
  write_image(a, fset, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, rt.image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  Pipeline fill = make_compute(a, sh, "csFill", {csl}, "guest_cs_fill");
  Pipeline smooth = make_compute(a, sh, "csSmooth", {csl}, "guest_cs_smooth");
  VkDescriptorSetLayout empty = make_set_layout(a, {});
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {empty, fsl}; gd.colors = {a.swap_format};
  Pipeline present = make_graphics(a, gd, "guest_draw");

  const VkAccessFlags2 storage = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    // Rasterizer::DispatchDirect -> BindResources: storage image -> GENERAL, SHADER_READ | SHADER_WRITE
    transit(c, rt, VK_IMAGE_LAYOUT_GENERAL, storage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, fill.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, fill.layout, 0, 1, &cset, 0, nullptr);
    vkCmdDispatch(c, 8, 8, 1);
    transit(c, rt, VK_IMAGE_LAYOUT_GENERAL, storage, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, smooth.pipe);
    vkCmdDispatch(c, 8, 8, 1);
    // Rasterizer::Draw -> sampled image -> SHADER_READ_ONLY_OPTIMAL
    transit(c, rt, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, present.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, present.layout, 1, 1, &fset, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
