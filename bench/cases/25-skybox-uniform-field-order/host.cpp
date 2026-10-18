// Distilled from bevyengine/bevy PR #18863: the host-side SkyboxUniforms put
// `transform` (mat4) before `brightness` (f32), while the shader declared
// brightness first. The mat4's 16-byte alignment makes the shader insert
// padding after brightness, so on WebGL2 (std140 uniforms) every field was
// read from the wrong offset. The fix reorders the shader struct to match:
// transform first, then brightness and the WebGL2 padding words.
#include "../../common/app.hpp"
using namespace bench;

struct SkyboxUniforms {
  float transform[16];
  float brightness;
  uint32_t _webgl2_padding_8b, _webgl2_padding_12b, _webgl2_padding_16b;
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Buffer ubo = make_buffer(a, sizeof(SkyboxUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "skybox_uniforms");
  Image cube{}; cube.format = VK_FORMAT_R16G16B16A16_SFLOAT; cube.w = cube.h = 128; cube.mips = 1;
  {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; ii.imageType = VK_IMAGE_TYPE_2D; ii.format = cube.format;
    ii.extent = {128, 128, 1}; ii.mipLevels = 1; ii.arrayLayers = 6; ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    VK(vkCreateImage(a.dev, &ii, nullptr, &cube.img));
    VkMemoryRequirements r; vkGetImageMemoryRequirements(a.dev, cube.img, &r);
    VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mi.allocationSize = r.size; mi.memoryTypeIndex = mem_type(a, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK(vkAllocateMemory(a.dev, &mi, nullptr, &cube.mem));
    VK(vkBindImageMemory(a.dev, cube.img, cube.mem, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = cube.img; vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE; vi.format = cube.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
    VK(vkCreateImageView(a.dev, &vi, nullptr, &cube.view));
    a.name(cube.img, VK_OBJECT_TYPE_IMAGE, "skybox_cubemap");
    a.name(cube.view, VK_OBJECT_TYPE_IMAGE_VIEW, "skybox_cubemap");
  }
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "skybox_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, cube.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline pipe = make_graphics(a, gd, "skybox");
  SkyboxUniforms u{};
  for (int i = 0; i < 4; i++) u.transform[i * 5] = 1;
  u.brightness = 1000.0f / 1000.0f;
  memcpy(ubo.map, &u, sizeof u);
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (f == 0)
    {
      VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT; b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
      b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
      b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; b.image = cube.img;
      b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
      VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
      vkCmdPipelineBarrier2(c, &di);
    }
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
