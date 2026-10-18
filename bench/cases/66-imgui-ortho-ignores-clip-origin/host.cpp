// Distilled from ocornut/imgui PR #3146 (fixes #3143): with
// glClipControl(GL_UPPER_LEFT, ...) the OpenGL clip-space y axis points down,
// as in Vulkan and D3D. ImGui_ImplOpenGL3_SetupRenderState() built its
// orthographic projection for the default lower-left origin:
//   L = DisplayPos.x, R = L + DisplaySize.x, T = DisplayPos.y, B = T + DisplaySize.y
//   ortho = { 2/(R-L),0,0,0 }, { 0,2/(T-B),0,0 }, { 0,0,-1,0 }, { (R+L)/(L-R),(T+B)/(B-T),0,1 }
// so display y = T (top) went to clip y = +1, which with an upper-left clip
// origin is the bottom of the framebuffer: the UI rendered upside down. The
// fix queries GL_CLIP_ORIGIN and, for an upper-left origin, swaps T and B
// ("Swap top and bottom if origin is upper left"). Port: the backend draws
// through Vulkan, whose clip origin is always upper-left (the case
// glClipControl(GL_UPPER_LEFT) selects); clip_origin_lower_left stands for the
// GL_CLIP_ORIGIN query.
#include "../../common/app.hpp"
using namespace bench;

struct ImDrawVert { float pos[2]; float uv[2]; uint32_t col; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  gd.vertex_stride = sizeof(ImDrawVert);
  gd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32_SFLOAT, 8}, {2, 0, VK_FORMAT_R8G8B8A8_UNORM, 16}};
  Pipeline shader_handle = make_graphics(a, gd, "imgui_opengl3_program");

  Image font = make_image(a, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "FontTexture");
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  image_barrier(up, font.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkClearColorValue white{{1, 1, 1, 1}}; VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(up, font.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &rng);
  image_barrier(up, font.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  Buffer ubo = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "ProjMtx");
  auto set = alloc_set(a, setl, "imgui_uniforms");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, font.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  // a window title bar at the top of the display, the window body below it
  Buffer vb = make_buffer(a, 12 * sizeof(ImDrawVert), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "ImDrawList.VtxBuffer");
  ImDrawVert v[12] = {
      {{40, 40}, {0, 0}, 0xff7a4a29u}, {{400, 40}, {0, 0}, 0xff7a4a29u}, {{40, 60}, {0, 0}, 0xff7a4a29u},
      {{400, 40}, {0, 0}, 0xff7a4a29u}, {{400, 60}, {0, 0}, 0xff7a4a29u}, {{40, 60}, {0, 0}, 0xff7a4a29u},
      {{40, 60}, {0, 0}, 0xf0242424u}, {{400, 60}, {0, 0}, 0xf0242424u}, {{40, 300}, {0, 0}, 0xf0242424u},
      {{400, 60}, {0, 0}, 0xf0242424u}, {{400, 300}, {0, 0}, 0xf0242424u}, {{40, 300}, {0, 0}, 0xf0242424u}};
  memcpy(vb.map, v, sizeof v);

  const bool clip_origin_lower_left = false;   // glGetIntegerv(GL_CLIP_ORIGIN) == GL_UPPER_LEFT
  const float DisplayPos[2] = {0, 0}, DisplaySize[2] = {(float)a.extent.width, (float)a.extent.height};

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, shader_handle.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, shader_handle.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdDraw(c, 12, 1, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
    // ImGui_ImplOpenGL3_SetupRenderState()
    float L = DisplayPos[0];
    float R = DisplayPos[0] + DisplaySize[0];
    float T = DisplayPos[1];
    float B = DisplayPos[1] + DisplaySize[1];
#ifdef FIXED
    if (!clip_origin_lower_left) { float tmp = T; T = B; B = tmp; }   // Swap top and bottom if origin is upper left
#else
    (void)clip_origin_lower_left;
#endif
    const float ortho_projection[4][4] = {
        {2.0f / (R - L), 0.0f, 0.0f, 0.0f},
        {0.0f, 2.0f / (T - B), 0.0f, 0.0f},
        {0.0f, 0.0f, -1.0f, 0.0f},
        {(R + L) / (L - R), (T + B) / (B - T), 0.0f, 1.0f},
    };
    memcpy(ubo.map, ortho_projection, sizeof ortho_projection);   // glUniformMatrix4fv(ProjMtx, 1, GL_FALSE, ...)
  });
}
