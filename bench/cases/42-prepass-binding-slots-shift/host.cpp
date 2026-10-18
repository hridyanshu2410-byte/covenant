// Distilled from bevyengine/bevy PR #10788: the mesh view bind group layout
// gets its prepass texture entries from get_bind_group_layout_entries(),
// which returned a SmallVec holding only the enabled prepass textures (depth,
// normal, motion vectors, deferred), and the caller zipped that list with the
// fixed binding indices [17, 18, 19, 20]. With depth and motion-vector
// prepasses but no normal prepass, the motion-vector entry was declared at
// 18, the normal slot, while the shader (and the bind group) use 19. The fix
// returns [Option<BindGroupLayoutEntryBuilder>; 4] with each texture at its
// own position and only adds the present ones, at their own index.
#include "../../common/app.hpp"
#include <array>
#include <optional>
using namespace bench;

enum PrepassFlags : uint32_t { DEPTH_PREPASS = 1, NORMAL_PREPASS = 2, MOTION_VECTOR_PREPASS = 4, DEFERRED_PREPASS = 8 };
static const uint32_t PREPASS_BINDINGS[4] = {17, 18, 19, 20};

static Binding texture_entry() {
  return {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT};
}

#ifdef FIXED
static std::array<std::optional<Binding>, 4> get_bind_group_layout_entries(uint32_t flags) {
  std::array<std::optional<Binding>, 4> entries{};
  if (flags & DEPTH_PREPASS) entries[0] = texture_entry();
  if (flags & NORMAL_PREPASS) entries[1] = texture_entry();
  if (flags & MOTION_VECTOR_PREPASS) entries[2] = texture_entry();
  if (flags & DEFERRED_PREPASS) entries[3] = texture_entry();
  return entries;
}
#else
static std::vector<Binding> get_bind_group_layout_entries(uint32_t flags) {
  std::vector<Binding> result;
  if (flags & DEPTH_PREPASS) result.push_back(texture_entry());
  if (flags & NORMAL_PREPASS) result.push_back(texture_entry());
  if (flags & MOTION_VECTOR_PREPASS) result.push_back(texture_entry());
  if (flags & DEFERRED_PREPASS) result.push_back(texture_entry());
  return result;
}
#endif

static VkDescriptorSetLayout layout_mesh_view(App& a, uint32_t flags) {
  std::vector<Binding> entries = {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT}};
  auto prepass = get_bind_group_layout_entries(flags);
#ifdef FIXED
  for (size_t i = 0; i < prepass.size(); i++)
    if (prepass[i]) { Binding b = *prepass[i]; b.binding = PREPASS_BINDINGS[i]; entries.push_back(b); }
#else
  for (size_t i = 0; i < prepass.size() && i < 4; i++) {          // zip([17, 18, 19, 20], entries)
    Binding b = prepass[i]; b.binding = PREPASS_BINDINGS[i]; entries.push_back(b);
  }
#endif
  return make_set_layout(a, entries);
}

struct ViewUniform { float view_proj[16]; float viewport[4]; };

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  const uint32_t flags = DEPTH_PREPASS | MOTION_VECTOR_PREPASS;     // camera with depth + motion vector prepass
  Image prepass_depth = make_image(a, 640, 480, VK_FORMAT_D32_SFLOAT,
                                   VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "prepass_depth");
  Image motion_vectors = make_image(a, 640, 480, VK_FORMAT_R16G16_SFLOAT,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "prepass_motion_vectors");
  Buffer view = make_buffer(a, sizeof(ViewUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "view_uniforms");
  ViewUniform vu{}; for (int i = 0; i < 4; i++) vu.view_proj[i * 5] = 1; vu.viewport[2] = 640; vu.viewport[3] = 480;
  memcpy(view.map, &vu, sizeof vu);

  VkDescriptorSetLayout view_layout = layout_mesh_view(a, flags);
  VkDescriptorSet view_bind_group = alloc_set(a, view_layout, "mesh_view_bind_group");
  // bind group entries: each prepass texture at its own binding
  write_buffer(a, view_bind_group, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, view);
  write_image(a, view_bind_group, 17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, prepass_depth.view,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_image(a, view_bind_group, 19, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, motion_vectors.view,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {view_layout}; gd.colors = {a.swap_format};
  Pipeline pbr = make_graphics(a, gd, "pbr_opaque_mesh");

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0) {
      depth_barrier(c, prepass_depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
      image_barrier(c, motion_vectors.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr.layout, 0, 1, &view_bind_group, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
