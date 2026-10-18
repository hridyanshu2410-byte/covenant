// Distilled from pixijs/pixijs PR #12207 ("fix: update the WebGPU front face
// when a reused pass changes flipY", merged 2026-09-16). PixiJS flips the
// projection when rendering into a texture with flipY, and inverts the
// pipeline's front face to match (GpuStateSystem / PipelineSystem key the
// pipeline on the target's resolved winding). In
// GpuRenderTargetAdaptor.startRenderPass(), binding the same target again with
// no clear reuses the open render pass and returned early -- before
// this._renderer.pipeline.setRenderTarget(renderTarget). A rebind that only
// changed flipY therefore drew with the flipped projection but with
// pipelines still keyed to the old front face, so back-face culling removed
// the triangles that should be visible and kept the ones that should be culled.
// The fix calls pipeline.setRenderTarget() before the reuse check.
// Port: WebGPU -> Vulkan; the pipeline cache keeps one pipeline per front
// face, the global uniform group is one UBO per bind, the render texture is
// shown on the swapchain by a full-screen pass. The scene is the PR's visual
// test: one CCW (front) and one CW (back) triangle with culling on, drawn into
// the same texture first with flipY = false, then again with flipY = true.
#include "../../common/app.hpp"
using namespace bench;

struct Mat3Std140 { float c[3][4]; };   // column-major float3x3, std140 columns padded to vec4
static Mat3Std140 mat3(float a, float b, float c, float d, float tx, float ty) {   // pixi Matrix (a b c d tx ty)
  Mat3Std140 m{}; m.c[0][0] = a; m.c[0][1] = b; m.c[1][0] = c; m.c[1][1] = d; m.c[2][0] = tx; m.c[2][1] = ty; m.c[2][2] = 1;
  return m;
}
struct GlobalUniforms { Mat3Std140 uProjectionMatrix; Mat3Std140 uWorldTransformMatrix; float uWorldColorAlpha[4]; float uResolution[2]; float _pad[2]; };
struct LocalUniforms { Mat3Std140 uTransformMatrix; float uColor[4]; };

struct RenderTarget { Image colorTexture; uint32_t width, height; };

// PipelineSystem: pipelines keyed by (program, state, front face of the bound target)
struct PipelineSystem {
  App* a; VkShaderModule sh; VkDescriptorSetLayout setl;
  Pipeline cache[2]{};          // [0] = ccw, [1] = cw
  bool frontFaceInverted = false;
  void setRenderTarget(bool targetFlipY) { frontFaceInverted = targetFlipY; }   // isFrontFaceInverted(target, flipY)
  Pipeline getPipeline() {
    int key = frontFaceInverted ? 1 : 0;
    if (!cache[key].pipe) cache[key] = create(key ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE);
    return cache[key];
  }
  Pipeline create(VkFrontFace ff) {
    Pipeline p{};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1; li.pSetLayouts = &setl;
    VK(vkCreatePipelineLayout(a->dev, &li, nullptr, &p.layout));
    VkPipelineShaderStageCreateInfo st[2]{};
    st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = sh; st[0].pName = "meshVertex";
    st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = sh; st[1].pName = "meshFragment";
    VkVertexInputBindingDescription vb[2] = {{0, 8, VK_VERTEX_INPUT_RATE_VERTEX}, {1, 8, VK_VERTEX_INPUT_RATE_VERTEX}};
    VkVertexInputAttributeDescription at[2] = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}, {1, 1, VK_FORMAT_R32G32_SFLOAT, 0}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 2; vi.pVertexBindingDescriptions = vb;
    vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = at;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_BACK_BIT; rs.frontFace = ff;    // state.culling = true
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cb{}; cb.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = 1; cbs.pAttachments = &cb;
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
    VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rinfo.colorAttachmentCount = 1; rinfo.pColorAttachmentFormats = &cf;
    VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gi.pNext = &rinfo; gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
    gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
    gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = p.layout;
    VK(vkCreateGraphicsPipelines(a->dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p.pipe));
    a->name(p.pipe, VK_OBJECT_TYPE_PIPELINE, ff == VK_FRONT_FACE_CLOCKWISE ? "mesh pipeline (cw)" : "mesh pipeline (ccw)");
    return p;
  }
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t size = 128;

  RenderTarget target{make_image(a, size, size, VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "renderTexture"), size, size};

  // MeshGeometry: points [0,0, 1,0, 0,1, 1,1] * size, indices 2,1,0 (front) and 2,1,3 (back)
  const float s = 56.f;
  const float positions[8] = {0, 0, s, 0, 0, s, s, s};
  const float uvs[8] = {0, 0, 1, 0, 0, 1, 1, 1};
  const uint32_t indices[6] = {2, 1, 0, 2, 1, 3};
  Buffer posBuf = make_buffer(a, sizeof positions, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "aPosition");
  Buffer uvBuf = make_buffer(a, sizeof uvs, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "aUV");
  Buffer idxBuf = make_buffer(a, sizeof indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, "indexBuffer");
  memcpy(posBuf.map, positions, sizeof positions); memcpy(uvBuf.map, uvs, sizeof uvs); memcpy(idxBuf.map, indices, sizeof indices);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT}});
  // GlobalUniformSystem.push({ projection from the bound target, flipY }) -> one uniform group per bind
  auto makeGlobals = [&](bool flipY, const std::string& nm) {
    GlobalUniforms g{};
    // calculateProjection(pm, 0, 0, width, height, flipY): pixels -> clip
    float sy = flipY ? -1.f : 1.f;
    g.uProjectionMatrix = mat3(2.f / size, 0, 0, sy * 2.f / size, -1, -sy);
    g.uWorldTransformMatrix = mat3(1, 0, 0, 1, 0, 0);
    g.uWorldColorAlpha[0] = g.uWorldColorAlpha[1] = g.uWorldColorAlpha[2] = g.uWorldColorAlpha[3] = 1;
    g.uResolution[0] = g.uResolution[1] = (float)size;
    Buffer b = make_buffer(a, sizeof g, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, nm);
    memcpy(b.map, &g, sizeof g);
    return b;
  };
  Buffer globals0 = makeGlobals(false, "globalUniforms (flipY=false)");
  Buffer globals1 = makeGlobals(true, "globalUniforms (flipY=true)");
  // two meshes: red at (4,4), green at (68,68) (mesh.position)
  auto makeLocal = [&](float x, float y, float r, float g, const std::string& nm) {
    LocalUniforms l{}; l.uTransformMatrix = mat3(1, 0, 0, 1, x, y); l.uColor[0] = r; l.uColor[1] = g; l.uColor[2] = 0; l.uColor[3] = 1;
    Buffer b = make_buffer(a, sizeof l, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, nm);
    memcpy(b.map, &l, sizeof l);
    return b;
  };
  Buffer local0 = makeLocal(4, 4, 1, 0, "localUniforms (mesh 0)");
  Buffer local1 = makeLocal(68, 68, 0, 1, "localUniforms (mesh 1)");
  VkDescriptorSet set0 = alloc_set(a, setl, "bindGroup (pass flipY=false)");
  write_buffer(a, set0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, globals0);
  write_buffer(a, set0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, local0);
  VkDescriptorSet set1 = alloc_set(a, setl, "bindGroup (rebind flipY=true)");
  write_buffer(a, set1, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, globals1);
  write_buffer(a, set1, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, local1);

  PipelineSystem pipeline{&a, sh, setl};

  auto blitl = make_set_layout(a, {{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto blitSet = alloc_set(a, blitl, "sprite bindGroup");
  write_image(a, blitSet, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, target.colorTexture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc bd; bd.vs = bd.fs = sh; bd.vs_entry = "blitVertex"; bd.fs_entry = "blitFragment"; bd.sets = {blitl}; bd.colors = {a.swap_format};
  Pipeline blit = make_graphics(a, bd, "sprite pipeline");

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    auto drawMesh = [&](VkDescriptorSet set) {
      Pipeline p = pipeline.getPipeline();
      vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &set, 0, nullptr);
      VkBuffer bufs[2] = {posBuf.buf, uvBuf.buf}; VkDeviceSize offs[2] = {0, 0};
      vkCmdBindVertexBuffers(c, 0, 2, bufs, offs);
      vkCmdBindIndexBuffer(c, idxBuf.buf, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(c, 6, 1, 0, 0, 0);
    };
    image_barrier(c, target.colorTexture.img, f == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    // renderTarget.bind({ target, clear: true, flipY: false }) -> startRenderPass
    pipeline.setRenderTarget(false);
    VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    att.imageView = target.colorTexture.view; att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {size, size}}; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &att;
    vkCmdBeginRendering(c, &ri);
    VkViewport vp{0, 0, (float)size, (float)size, 0, 1}; VkRect2D sc{{0, 0}, {size, size}};
    vkCmdSetViewport(c, 0, 1, &vp); vkCmdSetScissor(c, 0, 1, &sc);
    drawMesh(set0);
    // renderTarget.bind({ target, clear: false, flipY: true }): same target, no clear
    // -> the open render pass is reused
#ifdef FIXED
    pipeline.setRenderTarget(true);   // key pipelines to the new winding before the reuse check
#endif
    vkCmdSetViewport(c, 0, 1, &vp);   // encoder.setViewport(viewport); return
    drawMesh(set1);
    vkCmdEndRendering(c);
    image_barrier(c, target.colorTexture.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    // the sprite showing the texture on screen
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, blit.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, blit.layout, 0, 1, &blitSet, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
  return 0;
}
