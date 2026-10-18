// Distilled from KhronosGroup/Vulkan-Tutorial PR #484 ("fix/377-gltf-mirrored",
// fixes issue #377, merged 2026-09-03). Chapter 35 (glTF + KTX2) loaded every
// glTF position as
//   vertex.pos = {pos[0], -pos[1], pos[2]};   // "glTF is Y-up, Vulkan is Y-down"
// Negating one axis is a reflection, not a change of up axis: the model came
// out mirrored. To make the mirrored mesh visible the chapter also switched
// the pipeline to frontFace = Clockwise and used an initial rotation of -90
// degrees about X. The fix keeps the glTF positions as they are, restores
// frontFace = CounterClockwise (glTF winding, as in the other chapters), and
// rotates by +90 degrees about X to go from glTF's Y-up to the scene's Z-up.
// Port: vulkan-hpp RAII -> C API; tinygltf accessor reads are kept, the glTF
// buffer is an embedded asymmetric mesh (an "L" shaped prism, so a mirror is
// visible), the base-colour texture is replaced by vertex colours.
#include "../../common/app.hpp"
#include <cmath>
using namespace bench;

struct Vertex { float pos[3]; float color[3]; float texCoord[2]; };
struct UniformBufferObject { float model[16], view[16], proj[16]; };

// ---- glm stand-ins (column-major) ----
static void mat_identity(float* m) { for (int i = 0; i < 16; i++) m[i] = (i % 5 == 0) ? 1.f : 0.f; }
static void mat_mul(float* r, const float* a, const float* b) {
  float t[16];
  for (int c = 0; c < 4; c++) for (int rr = 0; rr < 4; rr++) {
    float s = 0; for (int k = 0; k < 4; k++) s += a[k * 4 + rr] * b[c * 4 + k]; t[c * 4 + rr] = s; }
  memcpy(r, t, sizeof t);
}
static void rotate(float* m, float angle, int axis) {   // glm::rotate(mat4(1), angle, unit axis)
  mat_identity(m); float c = cosf(angle), s = sinf(angle);
  int i = (axis + 1) % 3, j = (axis + 2) % 3;
  m[i * 4 + i] = c; m[i * 4 + j] = s; m[j * 4 + i] = -s; m[j * 4 + j] = c;
}
static void look_at(float* m, const float e[3], const float c[3], const float up[3]) {
  float f[3] = {c[0] - e[0], c[1] - e[1], c[2] - e[2]};
  float fl = sqrtf(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]); for (float& v : f) v /= fl;
  float s[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]};
  float sl = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]); for (float& v : s) v /= sl;
  float u[3] = {s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]};
  mat_identity(m);
  m[0] = s[0]; m[4] = s[1]; m[8] = s[2]; m[1] = u[0]; m[5] = u[1]; m[9] = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[12] = -(s[0] * e[0] + s[1] * e[1] + s[2] * e[2]);
  m[13] = -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]);
  m[14] = (f[0] * e[0] + f[1] * e[1] + f[2] * e[2]);
}
static void perspective(float* m, float fovy, float aspect, float n, float f) {   // GLM_FORCE_DEPTH_ZERO_TO_ONE
  for (int i = 0; i < 16; i++) m[i] = 0;
  float t = tanf(fovy / 2);
  m[0] = 1 / (aspect * t); m[5] = 1 / t; m[10] = f / (n - f); m[11] = -1; m[14] = -(f * n) / (f - n);
}

// ---- embedded glTF primitive: POSITION (VEC3 float), TEXCOORD_0, indices (u16), CCW ----
static void build_gltf_buffer(std::vector<float>& positions, std::vector<float>& uvs, std::vector<uint16_t>& indices) {
  // an "L" shaped extruded profile in the XY plane (glTF Y-up), extruded along +Z
  const float prof[6][2] = {{0, 0}, {0.6f, 0}, {0.6f, 0.2f}, {0.2f, 0.2f}, {0.2f, 0.8f}, {0, 0.8f}};
  for (int side = 0; side < 2; side++)
    for (auto& p : prof) { positions.insert(positions.end(), {p[0] - 0.3f, p[1] - 0.4f, side ? 0.15f : -0.15f}); uvs.insert(uvs.end(), {p[0], p[1]}); }
  // front cap (+Z, CCW seen from +Z) and back cap
  const uint16_t cap[12] = {0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5};
  for (int k = 0; k < 12; k += 3) indices.insert(indices.end(), {(uint16_t)(cap[k] + 6), (uint16_t)(cap[k + 1] + 6), (uint16_t)(cap[k + 2] + 6)});
  for (int k = 0; k < 12; k += 3) indices.insert(indices.end(), {cap[k], cap[k + 2], cap[k + 1]});
  for (uint16_t i = 0; i < 6; i++) {   // side walls, outward facing
    uint16_t j = (i + 1) % 6;
    indices.insert(indices.end(), {i, j, (uint16_t)(j + 6), i, (uint16_t)(j + 6), (uint16_t)(i + 6)});
  }
}

static Pipeline createGraphicsPipeline(App& a, VkShaderModule m, VkDescriptorSetLayout setl, VkFormat depthFormat) {
  Pipeline p{};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1; li.pSetLayouts = &setl;
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  st[0].module = m; st[0].pName = "vertMain";
  st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  st[1].module = m; st[1].pName = "fragMain";
  VkVertexInputBindingDescription vb{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription at[3] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
                                             {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
                                             {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texCoord)}};
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb;
  vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = at;
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL; rs.lineWidth = 1.0f;
  rs.cullMode = VK_CULL_MODE_BACK_BIT;                     // Re-enabled culling for better performance
#ifdef FIXED
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;          // glTF uses counter-clockwise winding
#else
  rs.frontFace = VK_FRONT_FACE_CLOCKWISE;                  // Keeping Clockwise for glTF
#endif
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS;
  VkPipelineColorBlendAttachmentState cb{}; cb.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1; cbs.pAttachments = &cb;
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rinfo.colorAttachmentCount = 1; rinfo.pColorAttachmentFormats = &a.swap_format; rinfo.depthAttachmentFormat = depthFormat;
  VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gi.pNext = &rinfo; gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
  gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
  gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = p.layout;
  VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p.pipe));
  a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, "graphicsPipeline");
  return p;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
  Image depth = make_image(a, a.extent.width, a.extent.height, depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "depthImage");

  // loadModel(): read the primitive's accessors
  std::vector<float> posData, uvData; std::vector<uint16_t> indexData;
  build_gltf_buffer(posData, uvData, indexData);
  std::vector<Vertex> vertices;
  for (size_t i = 0; i < posData.size() / 3; i++) {
    Vertex vertex{};
    const float* pos = &posData[i * 3];
#ifdef FIXED
    vertex.pos[0] = pos[0]; vertex.pos[1] = pos[1]; vertex.pos[2] = pos[2];
#else
    // glTF uses a right-handed coordinate system with Y-up
    // Vulkan uses a right-handed coordinate system with Y-down
    // We need to flip the Y coordinate
    vertex.pos[0] = pos[0]; vertex.pos[1] = -pos[1]; vertex.pos[2] = pos[2];
#endif
    const float* tc = &uvData[i * 2];
    vertex.texCoord[0] = tc[0]; vertex.texCoord[1] = tc[1];
    vertex.color[0] = 0.8f; vertex.color[1] = 0.5f + 0.5f * tc[1]; vertex.color[2] = 0.3f + 0.5f * tc[0];
    vertices.push_back(vertex);
  }
  Buffer vb = make_buffer(a, vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "vertexBuffer");
  memcpy(vb.map, vertices.data(), vertices.size() * sizeof(Vertex));
  Buffer ib = make_buffer(a, indexData.size() * 2, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, "indexBuffer");
  memcpy(ib.map, indexData.data(), indexData.size() * 2);

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  Buffer uniformBuffer = make_buffer(a, sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "uniformBuffers[0]");
  auto set = alloc_set(a, setl, "descriptorSets[0]");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformBuffer);
  Pipeline graphicsPipeline = createGraphicsPipeline(a, sh, setl, depthFormat);

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t imageIndex, int) {
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                  VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {a.swap_views[imageIndex]}, depth.view);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline.pipe);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    vkCmdBindIndexBuffer(c, ib.buf, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline.layout, 0, 1, &set, 0, nullptr);
    vkCmdDrawIndexed(c, (uint32_t)indexData.size(), 1, 0, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
    // updateUniformBuffer()
    float time = f * (1.0f / 60.0f);
    UniformBufferObject ubo{};
    float initialRotation[16], continuousRotation[16];
#ifdef FIXED
    rotate(initialRotation, 90.0f * 3.14159265f / 180.0f, 0);
#else
    rotate(initialRotation, -90.0f * 3.14159265f / 180.0f, 0);
#endif
    rotate(continuousRotation, time * 90.0f * 3.14159265f / 180.0f, 2);
    mat_mul(ubo.model, continuousRotation, initialRotation);
    const float eye[3] = {2, 2, 2}, center[3] = {0, 0, 0}, up[3] = {0, 0, 1};
    look_at(ubo.view, eye, center, up);
    perspective(ubo.proj, 45.0f * 3.14159265f / 180.0f, (float)a.extent.width / a.extent.height, 0.1f, 10.0f);
    memcpy(uniformBuffer.map, &ubo, sizeof ubo);
  });
  return 0;
}
