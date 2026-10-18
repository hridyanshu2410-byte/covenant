// Distilled from ghostty-org/ghostty PR #9695 ("fix(renderer): load
// linearized fg color for cursor cell", fixes #8353 / #9138, merged
// 2025-11-24). cell_text_vertex() always loads the glyph colour with
// load_color(..., linear = true) because cell_text_fragment() treats
// in.color as linear and, when linear blending is off (the default),
// re-encodes it with unlinearize(). For the text cell under the cursor, the
// vertex shader replaced the colour with uniforms.cursor_color loaded with
// linear = false, i.e. already sRGB-encoded; the fragment shader then
// encoded it a second time, so cursor text (cursor-text = cell-background
// and similar) came out too bright. The fix passes true.
// Port: Metal -> Slang/Vulkan. The Display P3 conversion is dropped (the
// Vulkan target is a UNORM swapchain holding sRGB-encoded values and
// use_display_p3 is false), uchar4 uniforms are packed into uints, the atlas
// is sampled with Load() at pixel coordinates, minimum-contrast handling and
// per-cell backgrounds are left out.
#include "../../common/app.hpp"
using namespace bench;

struct Uniforms {
  float projection_matrix[16];
  float screen_size[2];
  float cell_size[2];
  uint32_t grid_size[2];
  uint32_t _pad0[2];
  float grid_padding[4];
  float min_contrast;
  uint32_t _pad1;
  uint32_t cursor_pos[2];
  uint32_t cursor_color;   // uchar4
  uint32_t bg_color;       // uchar4
  uint32_t cursor_wide;
  uint32_t use_display_p3;
  uint32_t use_linear_blending;
  uint32_t use_linear_correction;
  uint32_t _pad2[2];
};
static_assert(offsetof(Uniforms, grid_padding) == 96 && offsetof(Uniforms, cursor_pos) == 120 &&
              offsetof(Uniforms, cursor_color) == 128 && offsetof(Uniforms, use_linear_correction) == 148, "std140");

struct CellText {
  uint32_t glyph_pos[2];
  uint32_t glyph_size[2];
  int16_t bearings[2];
  uint16_t grid_pos[2];
  uint8_t color[4];
  uint8_t atlas;
  uint8_t bools;
  uint8_t _pad[2];
};
enum { IS_CURSOR_GLYPH = 2 };

static uint32_t pack(uint8_t r, uint8_t g, uint8_t b, uint8_t a) { return r | g << 8 | b << 16 | (uint32_t)a << 24; }

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  App a = make_app(argc, argv, d);
  auto sh = load_shader(a, "shaders.spv");
  const float cw = 10, ch = 20;

  // grayscale glyph atlas: [0,10) a box glyph "A", [10,20) a full block (cursor)
  const uint32_t AW = 32, AH = 20;
  Image atlas = make_image(a, AW, AH, VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           "atlas_grayscale");
  Buffer staging = make_buffer(a, AW * AH, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "atlas_staging");
  uint8_t* px = (uint8_t*)staging.map;
  for (uint32_t y = 0; y < AH; y++)
    for (uint32_t x = 0; x < AW; x++)
      px[y * AW + x] = x < 10 ? ((x == 1 || x == 8 || y == 3 || y == 10) && y >= 3 && y < 17 ? 255 : 0) : (x < 20 ? 255 : 0);
  {
    VkCommandBuffer c = alloc_cmd(a, a.gfx_pool); begin(c);
    image_barrier(c, atlas.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {AW, AH, 1};
    vkCmdCopyBufferToImage(c, staging.buf, atlas.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    image_barrier(c, atlas.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    VK(vkEndCommandBuffer(c)); submit(a.gfx, c, {}, {}, VK_NULL_HANDLE); vkQueueWaitIdle(a.gfx);
  }

  // one row of text "AAAAA" (foreground #d8dee9) with a block cursor on column 2;
  // cursor-color = #d8dee9, cursor-text = cell-background (#2e3440)
  std::vector<CellText> cells;
  CellText cur{{10, 0}, {10, 20}, {0, 20}, {2, 0}, {0xd8, 0xde, 0xe9, 0xff}, 0, IS_CURSOR_GLYPH, {}};
  cells.push_back(cur);
  for (uint16_t x = 0; x < 5; x++)
    cells.push_back({{0, 0}, {10, 20}, {0, 20}, {x, 0}, {0xd8, 0xde, 0xe9, 0xff}, 0, 0, {}});
  Buffer cellBuf = make_buffer(a, cells.size() * sizeof(CellText), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "cells_text");
  memcpy(cellBuf.map, cells.data(), cells.size() * sizeof(CellText));

  Uniforms u{};
  // ortho(0, screen.w, 0, screen.h) with y down (Metal-style top-left origin)
  float W = (float)a.extent.width, H = (float)a.extent.height;
  float P[16] = {2 / W, 0, 0, 0, 0, 2 / H, 0, 0, 0, 0, 1, 0, -1, -1, 0, 1};
  memcpy(u.projection_matrix, P, sizeof P);
  u.screen_size[0] = W; u.screen_size[1] = H; u.cell_size[0] = cw; u.cell_size[1] = ch;
  u.grid_size[0] = (uint32_t)(W / cw); u.grid_size[1] = (uint32_t)(H / ch);
  u.min_contrast = 1.0f;
  u.cursor_pos[0] = 2; u.cursor_pos[1] = 0;
  u.cursor_color = pack(0x2e, 0x34, 0x40, 0xff);   // cursor text colour = cell background
  u.bg_color = pack(0x2e, 0x34, 0x40, 0xff);
  u.use_display_p3 = 0; u.use_linear_blending = 0; u.use_linear_correction = 0;
  Buffer ubo = make_buffer(a, sizeof u, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "uniforms");
  memcpy(ubo.map, &u, sizeof u);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "cell_text_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, atlas.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  // per-instance vertex data, 4-vertex triangle strip per glyph
  Pipeline pl{};
  {
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1; li.pSetLayouts = &setl;
    VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &pl.layout));
    VkPipelineShaderStageCreateInfo st[2]{};
    st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = sh; st[0].pName = "cell_text_vertex";
    st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = sh; st[1].pName = "cell_text_fragment";
    VkVertexInputBindingDescription vb{0, sizeof(CellText), VK_VERTEX_INPUT_RATE_INSTANCE};
    VkVertexInputAttributeDescription at[7] = {
        {0, 0, VK_FORMAT_R32G32_UINT, offsetof(CellText, glyph_pos)},
        {1, 0, VK_FORMAT_R32G32_UINT, offsetof(CellText, glyph_size)},
        {2, 0, VK_FORMAT_R16G16_SINT, offsetof(CellText, bearings)},
        {3, 0, VK_FORMAT_R16G16_UINT, offsetof(CellText, grid_pos)},
        {4, 0, VK_FORMAT_R8G8B8A8_UINT, offsetof(CellText, color)},
        {5, 0, VK_FORMAT_R8_UINT, offsetof(CellText, atlas)},
        {6, 0, VK_FORMAT_R8_UINT, offsetof(CellText, bools)}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb;
    vi.vertexAttributeDescriptionCount = 7; vi.pVertexAttributeDescriptions = at;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_NONE;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    // premultiplied alpha blending (ghostty's text pipeline)
    VkPipelineColorBlendAttachmentState cb{};
    cb.blendEnable = VK_TRUE; cb.colorWriteMask = 0xF;
    cb.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; cb.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
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
    gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = pl.layout;
    VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &pl.pipe));
    a.name(pl.pipe, VK_OBJECT_TYPE_PIPELINE, "cell_text");
  }

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pl.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pl.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0; vkCmdBindVertexBuffers(c, 0, 1, &cellBuf.buf, &off);
    vkCmdDraw(c, 4, (uint32_t)cells.size(), 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
