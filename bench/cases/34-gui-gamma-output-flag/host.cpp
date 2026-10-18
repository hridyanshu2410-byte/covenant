// Distilled from vchelaru/SkiaGameRendering #147 (fix PR #148): the
// premultiplied GUI material composites the GUI render texture onto the
// camera target. Its _GammaOutput flag skips the shader's sRGB decode, on
// the assumption that IMGUI writes gamma values even in a Linear project.
// That holds on Direct3D 11; on Metal the GUI texture's values are decoded
// like everything else, so skipping the decode passed encoded values to the
// sRGB framebuffer, which encoded them again (191 rendered as 224). The fix
// creates the material with gammaOutput only when the device is not Metal.
// This reproduction stands in for the Metal path.
#include "../../common/app.hpp"
using namespace bench;

enum class GraphicsDeviceType { Direct3D11, Metal };
static const GraphicsDeviceType kGraphicsDeviceType = GraphicsDeviceType::Metal;

struct PremultipliedParams { uint32_t gamma_output; float opacity; float pad[2]; };

struct Material { Buffer params; VkDescriptorSet set; };

static Material create_premultiplied_material(App& a, VkDescriptorSetLayout l, VkImageView gui, bool gamma_output) {
  Material m;
  m.params = make_buffer(a, sizeof(PremultipliedParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true,
                         "SkiaGameRendering Premultiplied GUI");
  PremultipliedParams p{gamma_output ? 1u : 0u, 1.0f, {0, 0}};
  memcpy(m.params.map, &p, sizeof p);
  m.set = alloc_set(a, l, "premultiplied_gui_set");
  write_image(a, m.set, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, gui, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  write_buffer(a, m.set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, m.params);
  return m;
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);                        // camera target: B8G8R8A8_SRGB (Linear project)
  // GUI render texture: 8-bit, holds the GUI's colours as authored (0,0,191 stays 191)
  Image gui = make_image(a, 256, 256, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "skia_gui_texture");
  Buffer staging = make_buffer(a, 256 * 256 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "gui_staging");
  uint8_t* px = (uint8_t*)staging.map;
  for (int i = 0; i < 256 * 256; i++) { px[i * 4 + 0] = 0; px[i * 4 + 1] = 0; px[i * 4 + 2] = 191; px[i * 4 + 3] = 255; }

  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
#ifdef FIXED
  bool gamma_output = kGraphicsDeviceType != GraphicsDeviceType::Metal;
#else
  bool gamma_output = true;     // IMGUI writes gamma values even in a Linear project
#endif
  Material gui_mat = create_premultiplied_material(a, setl, gui.view, gamma_output);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "premultiplied_gui");

  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0) {
      image_barrier(c, gui.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
      VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {256, 256, 1};
      vkCmdCopyBufferToImage(c, staging.buf, gui.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
      image_barrier(c, gui.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &gui_mat.set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
