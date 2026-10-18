// Distilled from vchelaru/SkiaGameRendering PR #117 (Stride adapter): Skia
// draws into a Vulkan texture through its own context and leaves the image in
// COLOR_ATTACHMENT_OPTIMAL. Stride's SpriteBatch then composites that
// texture; its descriptor says SHADER_READ_ONLY_OPTIMAL, but nothing moved
// the image there (VUID-vkCmdDraw-None-09600). The fix records a transition
// to the shader-resource state in the adapter's End() before compositing,
// and in Begin() moves the image back to the render-target layout with a
// small Stride submission so Skia's view of the layout matches next frame.
#include "../../common/app.hpp"
using namespace bench;

struct SkiaContext {        // stands in for GRDirectContext drawing into the surface
  App* a; Pipeline draw; VkCommandBuffer cmd; Image* surface; bool first = true;
  void flush_and_submit(int frame) {
    begin(cmd);
    if (first) {             // Skia's own initial transition
      image_barrier(cmd, surface->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      first = false;
    }
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = surface->view; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {surface->w, surface->h}}; ri.layerCount = 1;
    ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{0, 0, (float)surface->w, (float)surface->h, 0, 1}; VkRect2D sc{{0, 0}, {surface->w, surface->h}};
    vkCmdSetViewport(cmd, 0, 1, &vp); vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.pipe);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);       // Skia leaves the surface in COLOR_ATTACHMENT_OPTIMAL
    VK(vkEndCommandBuffer(cmd));
    VkFence f = make_fence(*a, false, "skia_flush_" + std::to_string(frame));
    submit(a->gfx, cmd, {}, {}, f);
    VK(vkWaitForFences(a->dev, 1, &f, VK_TRUE, UINT64_MAX));
  }
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Image surface = make_image(a, 512, 512, VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "skia_surface");
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc sd; sd.vs = sd.fs = sh; sd.vs_entry = "vsSkia"; sd.fs_entry = "fsSkia"; sd.colors = {VK_FORMAT_R8G8B8A8_UNORM};
  SkiaContext skia{&a, make_graphics(a, sd, "skia_draw"), alloc_cmd(a, a.gfx_pool), &surface};

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "spritebatch_texture");
  write_image(a, set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, surface.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vsSprite"; gd.fs_entry = "fsSprite"; gd.sets = {setl};
  gd.colors = {a.swap_format};
  Pipeline sprite = make_graphics(a, gd, "spritebatch");
  VkCommandBuffer stride_small = alloc_cmd(a, a.gfx_pool);

  run_frames(a, frames_from_env(4), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
#ifdef FIXED
    // SkiaStrideRenderTarget.End(): to ShaderResource before compositing
    image_barrier(c, surface.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
#endif
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, sprite.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, sprite.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
#ifdef FIXED
    // SkiaStrideRenderTarget.Begin(): back to RenderTarget with a small Stride submission
    if (f > 0) {
      begin(stride_small);
      image_barrier(stride_small, surface.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      VK(vkEndCommandBuffer(stride_small));
      VkFence fb = make_fence(a, false, "stride_begin_" + std::to_string(f));
      submit(a.gfx, stride_small, {}, {}, fb);
      VK(vkWaitForFences(a.dev, 1, &fb, VK_TRUE, UINT64_MAX));
    }
#endif
    skia.flush_and_submit(f);   // Skia draws the UI into the surface
  });
}
