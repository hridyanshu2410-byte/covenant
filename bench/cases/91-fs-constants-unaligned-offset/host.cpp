// Distilled from bkaradzic/bgfx PR #3663 ("D3D12: Align fragment shader
// constant buffer offset", merged 2026-04-15). In renderer_d3d12.cpp,
// commitShaderConstants() copied the vertex shader's constants to the
// per-draw scratch region and set the fragment shader's CBV to
// gpuAddress + m_vsh->m_size. A CBV's BufferLocation must be a multiple of
// D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT (256), and m_vsh->m_size
// (the VS constant block size) usually is not, so the fragment constants
// were read from the wrong place (debug layer error, garbage uniforms on
// some drivers). The fix uses
// bx::strideAlign(m_vsh->m_size, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT)
// as the FS offset in commitShaderConstants, the main submit loop and the
// blitter.
// Port: the scratch region is one host-visible buffer bound through two
// dynamic uniform-buffer bindings (VS at binding 0, FS at binding 1); the
// per-draw offsets play the role of the CBV addresses.
#include "../../common/app.hpp"
using namespace bench;

static uint32_t strideAlign(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }
constexpr uint32_t D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT = 256;

struct VsUniforms { float mvp[16], model[16], view[16], proj[16], viewRect[4]; };   // 272 bytes
struct FsUniforms { float tint[4], params[4]; };                                    // 32 bytes

struct ShaderD3D12 { uint32_t m_size; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  ShaderD3D12 vsh{sizeof(VsUniforms)}, fsh{sizeof(FsUniforms)};

  const uint32_t kDraws = 2, kRegion = 1024;
  Buffer scratch = make_buffer(a, kDraws * kRegion, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "constant_scratch");
  Buffer vb = make_buffer(a, 3 * 3 * sizeof(float), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "triangle_vb");
  float tri[9] = {-0.6f, -0.6f, 0, 0.6f, -0.6f, 0, 0, 0.6f, 0};
  memcpy(vb.map, tri, sizeof tri);

  // own pool: the bench pool has no dynamic uniform buffers
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4};
  VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pci.maxSets = 2; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
  VkDescriptorPool pool; VK(vkCreateDescriptorPool(a.dev, &pci, nullptr, &pool));
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_FRAGMENT_BIT}});
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &setl;
  VkDescriptorSet set; VK(vkAllocateDescriptorSets(a.dev, &ai, &set));
  a.name(set, VK_OBJECT_TYPE_DESCRIPTOR_SET, "shader_constants");
  VkDescriptorBufferInfo bi[2] = {{scratch.buf, 0, sizeof(VsUniforms)}, {scratch.buf, 0, sizeof(FsUniforms)}};
  VkWriteDescriptorSet w[2]{};
  for (int i = 0; i < 2; i++) {
    w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[i].dstSet = set; w[i].dstBinding = i; w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; w[i].pBufferInfo = &bi[i];
  }
  vkUpdateDescriptorSets(a.dev, 2, w, 0, nullptr);

  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  gd.vertex_stride = 12; gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
  Pipeline p = make_graphics(a, gd, "bgfx_program");

  // commitShaderConstants: VS block at the start of the draw's region, FS after it
#ifdef FIXED
  const uint32_t fsOffset = strideAlign(vsh.m_size, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
#else
  const uint32_t fsOffset = vsh.m_size;
#endif
  for (uint32_t d = 0; d < kDraws; d++) {
    uint8_t* region = (uint8_t*)scratch.map + d * kRegion;
    VsUniforms vs{};
    for (int k = 0; k < 4; k++) vs.mvp[k * 5] = vs.model[k * 5] = vs.view[k * 5] = vs.proj[k * 5] = 1;
    vs.mvp[12] = vs.model[12] = d ? 0.3f : -0.3f;
    vs.viewRect[2] = 640; vs.viewRect[3] = 480;
    FsUniforms fs{{d ? 0.2f : 0.9f, 0.5f, d ? 0.9f : 0.2f, 1}, {0.05f, 0, 0, 0}};
    memcpy(region, &vs, vsh.m_size);
    memcpy(region + fsOffset, &fs, fsh.m_size);
  }

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &zero);
    for (uint32_t d = 0; d < kDraws; d++) {
      uint32_t offs[2] = {d * kRegion, d * kRegion + fsOffset};
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &set, 2, offs);
      vkCmdDraw(c, 3, 1, 0, 0);
    }
    vkCmdEndRendering(c);
  });
  return 0;
}
