// Distilled from aether-lang-dev/ae3d #458 (fix PR #497): the last pass of
// a frame renders into an offscreen image that is both sampled by the next
// shader and copied to a host buffer for readback. The pass's outgoing
// (external) dependency covered only fragment-shader reads, not
// TRANSFER_READ, so the copy raced the pass (read-after-write), and the
// next frame's first pass transitioned the image while the previous
// readback could still be reading it (write-after-read). The fix: the
// pass's outgoing dependency also covers the transfer read, and the
// readback copy is followed by an execution barrier so the next frame's
// first pass waits for it before writing the image.
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  const int FRAMES_IN_FLIGHT = 2;
  Image frame_color = make_image(a, 640, 480, VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT, "frame_color");
  std::vector<Buffer> readback;
  for (int i = 0; i < FRAMES_IN_FLIGHT; i++)
    readback.push_back(make_buffer(a, 640 * 480 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                                   "readback[" + std::to_string(i) + "]"));
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc sd; sd.vs = sd.fs = sh; sd.vs_entry = "vsFullscreen"; sd.fs_entry = "fsScene"; sd.colors = {VK_FORMAT_R8G8B8A8_UNORM};
  Pipeline scene = make_graphics(a, sd, "scene_pass");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "present_set");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frame_color.view, VK_IMAGE_LAYOUT_GENERAL);
  GfxDesc pd; pd.vs = pd.fs = sh; pd.vs_entry = "vsFullscreen"; pd.fs_entry = "fsPresent"; pd.sets = {setl};
  pd.colors = {a.swap_format};
  Pipeline present_pass = make_graphics(a, pd, "present_pass");

  run_frames(a, frames_from_env(4), FRAMES_IN_FLIGHT, [&](VkCommandBuffer c, uint32_t idx, int f) {
    // the frame's pass: incoming dependency
    image_barrier(c, frame_color.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {frame_color.view});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.pipe);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    // outgoing (external) dependency, final layout GENERAL
#ifdef FIXED
    image_barrier(c, frame_color.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
#else
    image_barrier(c, frame_color.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
#endif
    // composite to the window
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, present_pass.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, present_pass.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    // readback of the frame
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {640, 480, 1};
    vkCmdCopyImageToBuffer(c, frame_color.img, VK_IMAGE_LAYOUT_GENERAL, readback[f % FRAMES_IN_FLIGHT].buf, 1, &r);
#ifdef FIXED
    {   // the next frame's first pass waits for the readback before writing the image
      VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
      mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; mb.srcAccessMask = 0;
      mb.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT; mb.dstAccessMask = 0;
      VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
      vkCmdPipelineBarrier2(c, &di);
    }
#endif
  });
}
