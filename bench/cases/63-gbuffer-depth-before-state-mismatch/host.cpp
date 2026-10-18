// Distilled from shuhuai/DeferredShadingD3D12 #2 (fix described in the issue;
// D3D12 resource states ported to Vulkan image layouts). The depth/stencil
// texture is created in D3D12_RESOURCE_STATE_DEPTH_WRITE and written by the
// G-buffer pass. DeferredRender::ApplyLightingPSO() then issued
//   AddResourceBarrier(mRtvTexture[i], RENDER_TARGET, GENERIC_READ)  (each RT)
//   AddResourceBarrier(mDsTexture, DEPTH_READ, GENERIC_READ)
// and drew the light pass. The depth barrier's "before" state (DEPTH_READ)
// is not the texture's state (DEPTH_WRITE), and nothing moves the G-buffer
// targets or the depth texture back, so the next frame's G-buffer pass
// renders into resources left in GENERIC_READ and the next barrier's before
// state no longer matches either (debug layer error #527
// RESOURCE_BARRIER_BEFORE_AFTER_MISMATCH). The fix in the issue transitions
// the depth texture from DEPTH_WRITE, and after the light pass transitions
// depth GENERIC_READ -> DEPTH_WRITE and every RT GENERIC_READ ->
// RENDER_TARGET. Port: RENDER_TARGET = COLOR_ATTACHMENT_OPTIMAL,
// DEPTH_WRITE = DEPTH_STENCIL_ATTACHMENT_OPTIMAL, DEPTH_READ =
// DEPTH_STENCIL_READ_ONLY_OPTIMAL, GENERIC_READ = SHADER_READ_ONLY_OPTIMAL;
// the stage/access masks are ours (D3D12 has none).
#include "../../common/app.hpp"
using namespace bench;

static const int numRTV = 3;

static void ds_barrier(VkCommandBuffer c, VkImage img, VkImageLayout from, VkImageLayout to,
                       VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.srcStageMask = ss; b.srcAccessMask = sa; b.dstStageMask = ds; b.dstAccessMask = da;
  b.oldLayout = from; b.newLayout = to; b.image = img;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
  vkCmdPipelineBarrier2(c, &di);
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const VkFormat rtFormats[numRTV] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM};
  const char* rtNames[numRTV] = {"mRtvTexture[0] (albedo)", "mRtvTexture[1] (normal)", "mRtvTexture[2] (specular)"};
  std::vector<Image> mRtvTexture;
  for (int i = 0; i < numRTV; i++)
    mRtvTexture.push_back(make_image(a, 640, 480, rtFormats[i], VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, rtNames[i]));
  Image mDsTexture = make_image(a, 640, 480, VK_FORMAT_D24_UNORM_S8_UINT,
                                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "mDsTexture");

  // CreateRTV / CreateDSV: resources are created in RENDER_TARGET / DEPTH_WRITE
  VkCommandBuffer init = alloc_cmd(a, a.gfx_pool);
  begin(init);
  for (auto& rt : mRtvTexture)
    image_barrier(init, rt.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  ds_barrier(init, mDsTexture.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
             VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
  VK(vkEndCommandBuffer(init));
  submit(a.gfx, init, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  Buffer cam = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera_cb");
  float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  memcpy(cam.map, id, sizeof id);
  auto camL = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto gbufL = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                   {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                   {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                   {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto camSet = alloc_set(a, camL, "m_cbvsrvHeap (camera)");
  write_buffer(a, camSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cam);
  auto gbufSet = alloc_set(a, gbufL, "m_cbvsrvHeap (gbuffer SRVs)");
  for (int i = 0; i < numRTV; i++)
    write_image(a, gbufSet, i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, mRtvTexture[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image(a, gbufSet, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, mDsTexture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vsGBuffer"; gd.fs_entry = "fsGBuffer"; gd.sets = {camL};
  gd.colors = {rtFormats[0], rtFormats[1], rtFormats[2]}; gd.depth = VK_FORMAT_D24_UNORM_S8_UINT;
  Pipeline m_pipelineState = make_graphics(a, gd, "m_pipelineState (gbuffer)");
  GfxDesc ld; ld.vs = ld.fs = sh; ld.vs_entry = "vsQuad"; ld.fs_entry = "fsLight"; ld.sets = {camL, gbufL};
  ld.colors = {a.swap_format};
  Pipeline mLightPso = make_graphics(a, ld, "mLightPso");

  run_frames(a, frames_from_env(4), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    // ApplyGBufferPSO + scene draw
    begin_rendering(a, c, {mRtvTexture[0].view, mRtvTexture[1].view, mRtvTexture[2].view}, mDsTexture.view);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineState.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineState.layout, 0, 1, &camSet, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);

    // ApplyLightingPSO
    for (int i = 0; i < numRTV; i++)
      image_barrier(c, mRtvTexture[i].img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
#ifdef FIXED
    ds_barrier(c, mDsTexture.img, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
#else
    ds_barrier(c, mDsTexture.img, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
#endif
               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mLightPso.pipe);
    VkDescriptorSet sets[2] = {camSet, gbufSet};
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mLightPso.layout, 0, 2, sets, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);                         // mQuadRender.Draw
    vkCmdEndRendering(c);
#ifdef FIXED
    ds_barrier(c, mDsTexture.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
               VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
               VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    for (int i = 0; i < numRTV; i++)
      image_barrier(c, mRtvTexture[i].img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
#endif
  });
}
