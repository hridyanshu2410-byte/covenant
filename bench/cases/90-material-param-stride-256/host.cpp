// Distilled from Force67/rx PR #39 ("renderer: fix material param stride
// overflow after skin SSS params", merged 2026-07-26). Material parameters
// live in one persistently mapped uniform buffer, one slot per material at
// param_index * kParamStride, bound with a dynamic offset. kParamStride was
// 256, which matched sizeof(MaterialSystem::Params) until the skin
// subsurface rows were appended and the struct became 272 bytes. Upload kept
// doing memcpy(mapped + param_index * kParamStride, &params, sizeof(params)),
// so every material's last 16 bytes spilled into the next slot, and the
// next material's upload overwrote the previous material's SSS row: skin
// materials took the neighbour's base colour as their scatter colour. The
// fix raises kParamStride to 512 and adds
// static_assert(sizeof(Params) <= kParamStride).
// Port: two materials (a skin material and a plain one) drawn with dynamic
// offsets; the struct body is reduced to float4 rows of the same size.
#include "../../common/app.hpp"
using namespace bench;

namespace MaterialSystem {
struct Params {
  float base_color[4];
  float emissive[4];
  float pbr[4];
  float uv_transform[4];
  float clearcoat[4];
  float sheen[4];
  float transmission[4];
  float anisotropy[4];
  float iridescence[4];
  uint32_t texture_indices0[4];
  uint32_t texture_indices1[4];
  uint32_t flags[4];
  float reserved[4][4];
  float sss_color_radius[4];      // added for skin SSS: 256 -> 272 bytes
};
}  // namespace MaterialSystem

#ifdef FIXED
constexpr VkDeviceSize kParamStride = 512;
static_assert(sizeof(MaterialSystem::Params) <= kParamStride, "material params exceed stride");
#else
constexpr VkDeviceSize kParamStride = 256;
#endif
constexpr uint32_t kMaxMaterials = 64;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");

  Buffer params_buf = make_buffer(a, kMaxMaterials * kParamStride, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true,
                                  "material_params");
  auto upload = [&](uint32_t param_index, const MaterialSystem::Params& params) {
    memcpy((uint8_t*)params_buf.map + param_index * kParamStride, &params, sizeof(params));
  };

  MaterialSystem::Params skin{}, cloth{};
  float sc[4] = {0.80f, 0.55f, 0.45f, 1}; memcpy(skin.base_color, sc, 16);
  float st[4] = {1, 1, 0, 0}; memcpy(skin.uv_transform, st, 16); memcpy(cloth.uv_transform, st, 16);
  float sss[4] = {1.0f, 0.35f, 0.25f, 2.5f}; memcpy(skin.sss_color_radius, sss, 16);
  skin.flags[0] = 1;                                               // skin shading
  float cc[4] = {0.10f, 0.20f, 0.70f, 1}; memcpy(cloth.base_color, cc, 16);
  upload(0, skin);
  upload(1, cloth);

  // own pool: the bench pool has no dynamic uniform buffers
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4};
  VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pci.maxSets = 4; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
  VkDescriptorPool pool; VK(vkCreateDescriptorPool(a.dev, &pci, nullptr, &pool));
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT}});
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &setl;
  VkDescriptorSet set; VK(vkAllocateDescriptorSets(a.dev, &ai, &set));
  a.name(set, VK_OBJECT_TYPE_DESCRIPTOR_SET, "material_params_set");
  VkDescriptorBufferInfo bi{params_buf.buf, 0, sizeof(MaterialSystem::Params)};
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet = set; w.dstBinding = 0; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  w.pBufferInfo = &bi;
  vkUpdateDescriptorSets(a.dev, 1, &w, 0, nullptr);

  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline mp = make_graphics(a, gd, "material_pass");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mp.pipe);
    for (uint32_t param_index = 0; param_index < 2; param_index++) {
      uint32_t off = (uint32_t)(param_index * kParamStride);
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mp.layout, 0, 1, &set, 1, &off);
      vkCmdDraw(c, 3, 1, 0, 0);
    }
    vkCmdEndRendering(c);
  });
  return 0;
}
