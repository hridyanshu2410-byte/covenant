// Distilled from Polyfrost/OneConfig PR #1102: OneConfig renders item icons
// into an offscreen atlas through the game's Vulkan backend, which renders
// to (and keeps) its colour targets in VK_IMAGE_LAYOUT_GENERAL, then samples
// the atlas in the GUI. NativeVulkanService's helpers disagreed with that:
//  - transitionOffscreenForRendering moved the atlas GENERAL ->
//    COLOR_ATTACHMENT_OPTIMAL, although the backend's pass then uses it as
//    GENERAL (image layout mismatch at submit), and transitionOffscreenFor
//    Sampling claimed oldLayout COLOR_ATTACHMENT_OPTIMAL;
//  - its first scope was FRAGMENT_SHADER / SHADER_READ only, so the second
//    isolated item render was not ordered after the first one's attachment
//    writes (WRITE_AFTER_WRITE against the previous storeOp).
// The fix keeps the atlas in GENERAL throughout and adds
// COLOR_ATTACHMENT_OUTPUT / COLOR_ATTACHMENT_WRITE to the render transition's
// first scope. (The PR's other half, an extra midFrameFlush submit, is not
// reproduced.)
#include "../../common/app.hpp"
using namespace bench;

// transitionImage(texture, oldLayout, newLayout, srcStage, dstStage, srcAccess, dstAccess) with sync1
static void transition_image(VkCommandBuffer c, VkImage img, VkImageLayout from, VkImageLayout to,
                             VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                             VkAccessFlags src_access, VkAccessFlags dst_access) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcAccessMask = src_access; b.dstAccessMask = dst_access; b.oldLayout = from; b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(c, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static void transition_offscreen_for_rendering(VkCommandBuffer c, VkImage img) {
#ifdef FIXED
  transition_image(c, img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT);
#else
  transition_image(c, img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_SHADER_READ_BIT,
                   VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT);
#endif
}

static void transition_offscreen_for_sampling(VkCommandBuffer c, VkImage img) {
#ifdef FIXED
  VkImageLayout old_layout = VK_IMAGE_LAYOUT_GENERAL;
#else
  VkImageLayout old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#endif
  transition_image(c, img, old_layout, VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                   VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}

// The game backend's render pass: colour target used in GENERAL, contents kept.
static void backend_begin_pass(VkCommandBuffer c, VkImageView view, VkRect2D area) {
  VkRenderingAttachmentInfo r{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  r.imageView = view; r.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  r.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; r.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = area; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &r;
  vkCmdBeginRendering(c, &ri);
  VkViewport vp{(float)area.offset.x, (float)area.offset.y, (float)area.extent.width, (float)area.extent.height, 0, 1};
  vkCmdSetViewport(c, 0, 1, &vp);
  vkCmdSetScissor(c, 0, 1, &area);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  Image atlas = make_image(a, 512, 512, VK_FORMAT_R8G8B8A8_UNORM,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "item_atlas");
  GfxDesc id; id.vs = id.fs = sh; id.vs_entry = "vsItem"; id.fs_entry = "fsItem";
  id.colors = {VK_FORMAT_R8G8B8A8_UNORM}; id.push_size = 16;
  Pipeline item = make_graphics(a, id, "item_icon");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "gui_atlas");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, atlas.view, VK_IMAGE_LAYOUT_GENERAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vsFullscreen"; gd.fs_entry = "fsGui"; gd.sets = {setl};
  gd.colors = {a.swap_format};
  Pipeline gui = make_graphics(a, gd, "gui");

  run_frames(a, frames_from_env(4), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0)   // the backend creates its colour textures in GENERAL
      transition_image(c, atlas.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    // MinecraftItemCatalogService: changed items are rendered in isolated batches
    for (int batch = 0; batch < 2; batch++) {
      transition_offscreen_for_rendering(c, atlas.img);
      VkRect2D cell{{batch * 256, 0}, {256, 256}};
      backend_begin_pass(c, atlas.view, cell);
      vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, item.pipe);
      float push[4] = {batch * 0.5f, 0.0f, 0.5f, 0.5f};
      vkCmdPushConstants(c, item.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof push, push);
      vkCmdDraw(c, 6, 1, 0, 0);
      vkCmdEndRendering(c);
    }
    transition_offscreen_for_sampling(c, atlas.img);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, gui.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, gui.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
