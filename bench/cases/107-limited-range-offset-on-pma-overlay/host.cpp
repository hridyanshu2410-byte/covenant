// Distilled from xbmc/xbmc PR #28386 (fixes issue #22383, merged 2026-06-05; fix
// commit 953cd6a "GLES: fix limited-range wash-out on premultiplied-alpha
// (PMA) bitmap overlays"). With the GUI set to limited-range output, Kodi's
// GLES texture shader converts every GUI texel to limited range with
//   rgb *= 219/255; rgb += 16/255;
// Bitmap subtitles (PGS, VobSub, DVD SPU, paletted BD-PG) are uploaded by
// COverlayTextureGLES as premultiplied-alpha textures and drawn with the PMA
// blend (GL_ONE, GL_ONE_MINUS_SRC_ALPHA). For a fully transparent PMA texel
// (rgb = 0, a = 0) the shader still emits rgb = 16/255, and with a source
// factor of ONE that offset is added to the video underneath, washing out the
// picture across the whole subtitle rectangle. The fix adds an m_pma uniform
// and scales the offset by alpha for PMA textures:
//   rgb.rgb += mix(1.0, rgb.a, m_pma) * 16.0 / 255.0;
// COverlayTextureGLES::Render sets m_pma = 1.0, CGLESShader::OnEnabled resets
// it to 0.0 for everyone else.
// Port: GLES 2 -> Slang/Vulkan; the video layer is a flat limited-range grey
// quad drawn first, the subtitle bitmap is a generated PMA texture.
#include "../../common/app.hpp"
using namespace bench;

struct GUIUniforms {
  float m_proj[16];
  float m_sdrPeak;
#ifdef FIXED
  float m_pma;
  float _pad[2];
#else
  float _pad[3];
#endif
};
struct GUIVertex { float pos[4]; float cord0[4]; };

static Pipeline make_gui_pipeline(App& a, VkShaderModule sh, VkDescriptorSetLayout setl, bool pmaBlend, const char* nm) {
  Pipeline p{};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1; li.pSetLayouts = &setl;
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  st[0].module = sh; st[0].pName = "vertMain";
  st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  st[1].module = sh; st[1].pName = "fragMain";
  VkVertexInputBindingDescription vb{0, sizeof(GUIVertex), VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription at[2] = {{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(GUIVertex, pos)},
                                             {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(GUIVertex, cord0)}};
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb;
  vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = at;
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_NONE;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineColorBlendAttachmentState cb{}; cb.colorWriteMask = 0xF;
  if (pmaBlend) {   // glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
    cb.blendEnable = VK_TRUE;
    cb.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; cb.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  }
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1; cbs.pAttachments = &cb;
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rinfo.colorAttachmentCount = 1; rinfo.pColorAttachmentFormats = &a.swap_format;
  VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gi.pNext = &rinfo; gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
  gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
  gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = p.layout;
  VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p.pipe));
  a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, nm);
  return p;
}

static Image upload_texture(App& a, uint32_t w, uint32_t h, const std::vector<uint8_t>& rgba, const std::string& nm) {
  Image t = make_image(a, w, h, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, nm);
  Buffer s = make_buffer(a, rgba.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, nm + " upload");
  memcpy(s.map, rgba.data(), rgba.size());
  VkCommandBuffer c = alloc_cmd(a, a.gfx_pool); begin(c);
  image_barrier(c, t.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {w, h, 1};
  vkCmdCopyBufferToImage(c, s.buf, t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
  image_barrier(c, t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  VK(vkEndCommandBuffer(c)); submit(a.gfx, c, {}, {}, VK_NULL_HANDLE); vkQueueWaitIdle(a.gfx);
  return t;
}

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;   // GLES default framebuffer (no sRGB)
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");
  const float W = (float)a.extent.width, H = (float)a.extent.height;

  // video frame stand-in: full-range mid grey (the limited-range conversion maps it)
  std::vector<uint8_t> video(4 * 4 * 4, 96);
  for (size_t i = 3; i < video.size(); i += 4) video[i] = 255;
  Image videoTex = upload_texture(a, 4, 4, video, "video_frame");
  // PGS subtitle bitmap, decoded from its palette and stored premultiplied (m_pma = true):
  // white text with a black outline in the middle rows, fully transparent elsewhere
  const uint32_t SW = 256, SH = 64;
  std::vector<uint8_t> sub(SW * SH * 4, 0);
  for (uint32_t y = 0; y < SH; y++)
    for (uint32_t x = 0; x < SW; x++) {
      uint8_t* p = &sub[(y * SW + x) * 4];
      bool glyph = y >= 24 && y < 40 && (x / 12) % 2 == 0 && x > 16 && x < 240;
      bool outline = !glyph && y >= 22 && y < 42 && x > 14 && x < 242 && ((x + 2) / 12) % 2 == 0;
      if (glyph) { p[0] = p[1] = p[2] = 235; p[3] = 235; }      // premultiplied: rgb = c * a
      else if (outline) { p[0] = p[1] = p[2] = 0; p[3] = 255; }
    }
  Image subTex = upload_texture(a, SW, SH, sub, "overlay_texture (PMA)");

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  // CGLESShader::OnEnabled() for each draw: projection, m_sdrPeak, (m_pma = 0)
  auto makeUniforms = [&](float pma, const std::string& nm) {
    (void)pma;
    GUIUniforms u{};
    float P[16] = {2 / W, 0, 0, 0, 0, 2 / H, 0, 0, 0, 0, 1, 0, -1, -1, 0, 1};
    memcpy(u.m_proj, P, sizeof P); u.m_sdrPeak = 1.0f;
#ifdef FIXED
    u.m_pma = pma;
#endif
    Buffer b = make_buffer(a, sizeof u, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, nm);
    memcpy(b.map, &u, sizeof u);
    return b;
  };
  Buffer videoUniforms = makeUniforms(0.0f, "gui uniforms (video)");
  // COverlayTextureGLES::Render: if (m_pma) glUniform1f(GUIShaderGetPma(), 1.0f)
  Buffer overlayUniforms = makeUniforms(1.0f, "gui uniforms (overlay)");
  VkDescriptorSet videoSet = alloc_set(a, setl, "video_set");
  write_buffer(a, videoSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, videoUniforms);
  write_image(a, videoSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, videoTex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  VkDescriptorSet overlaySet = alloc_set(a, setl, "overlay_set");
  write_buffer(a, overlaySet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, overlayUniforms);
  write_image(a, overlaySet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, subTex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  Pipeline opaque = make_gui_pipeline(a, sh, setl, false, "SM_TEXTURE_NOBLEND (video)");
  Pipeline pma = make_gui_pipeline(a, sh, setl, true, "SM_TEXTURE_NOBLEND (overlay, PMA blend)");

  // quads (triangle strip, idx {0, 1, 3, 2})
  auto quad = [](float x0, float y0, float x1, float y1) {
    return std::vector<GUIVertex>{{{x0, y0, 0, 1}, {0, 0, 0, 0}}, {{x1, y0, 0, 1}, {1, 0, 0, 0}},
                                  {{x0, y1, 0, 1}, {0, 1, 0, 0}}, {{x1, y1, 0, 1}, {1, 1, 0, 0}}};
  };
  auto vq = quad(0, 0, W, H), sq = quad(W / 2 - SW, H - 2 * SH - 20, W / 2 + SW, H - 20);
  Buffer vb = make_buffer(a, 8 * sizeof(GUIVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "gui vertices");
  memcpy(vb.map, vq.data(), 4 * sizeof(GUIVertex));
  memcpy((GUIVertex*)vb.map + 4, sq.data(), 4 * sizeof(GUIVertex));

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    VkDeviceSize off = 0; vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque.layout, 0, 1, &videoSet, 0, nullptr);
    vkCmdDraw(c, 4, 1, 0, 0);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pma.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pma.layout, 0, 1, &overlaySet, 0, nullptr);
    vkCmdDraw(c, 4, 1, 4, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
