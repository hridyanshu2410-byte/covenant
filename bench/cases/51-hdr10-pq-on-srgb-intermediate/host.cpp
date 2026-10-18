// Distilled from brianmk/coin PR #48: the raster path draws the scene into
// an RGBA16F intermediate and an output pass copies it to the swapchain.
// The visual/background shaders write display-referred sRGB code values (the
// SDR swapchain is plain UNORM and nothing linearizes in the scene pass).
// With an HDR10 swapchain (VK_FORMAT_A2B10G10R10_UNORM_PACK32,
// VK_COLOR_SPACE_HDR10_ST2084_EXT, BT.2020 primaries), the output pass's HDR
// branch treated the intermediate as linear BT.709 radiance: exposure, tone
// map and the PQ inverse EOTF were applied to sRGB-encoded values, which
// lifts every midtone (washed-out image), and the primaries were never
// converted to BT.2020. The fix decodes sRGB to linear (IEC 61966-2-1),
// converts BT.709 -> BT.2020, then applies exposure, tone map and PQ. The
// SDR branch is unchanged.
#include "../../common/app.hpp"
using namespace bench;

// Swapchain recreation for the HDR10 surface format the app selected.
static void recreate_swapchain(App& a, VkFormat fmt, VkColorSpaceKHR cs) {
  for (auto v : a.swap_views) vkDestroyImageView(a.dev, v, nullptr);
  a.swap_views.clear();
  VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sci.surface = a.surface; sci.minImageCount = 3; sci.imageFormat = fmt; sci.imageColorSpace = cs;
  sci.imageExtent = a.extent; sci.imageArrayLayers = 1; sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  sci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR; sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sci.presentMode = VK_PRESENT_MODE_FIFO_KHR; sci.clipped = VK_TRUE; sci.oldSwapchain = a.swapchain;
  VkSwapchainKHR sc; VK(vkCreateSwapchainKHR(a.dev, &sci, nullptr, &sc));
  vkDestroySwapchainKHR(a.dev, a.swapchain, nullptr);
  a.swapchain = sc; a.swap_format = fmt;
  a.name(sc, VK_OBJECT_TYPE_SWAPCHAIN_KHR, "swapchain_hdr10");
  uint32_t n = 0; vkGetSwapchainImagesKHR(a.dev, sc, &n, nullptr);
  a.swap_images.resize(n); vkGetSwapchainImagesKHR(a.dev, sc, &n, a.swap_images.data());
  for (auto img : a.swap_images) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v; VK(vkCreateImageView(a.dev, &vi, nullptr, &v)); a.swap_views.push_back(v);
  }
}

struct OutputParams { uint32_t hdr; float exposure; float paper_white_nits; float pad; };

int main(int argc, char** argv) {
  AppDesc d; d.swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  App a = make_app(argc, argv, d);
  recreate_swapchain(a, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT);
  auto sh = load_shader(a, "shaders.spv");
  Image scene = make_image(a, 640, 480, VK_FORMAT_R16G16B16A16_SFLOAT,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "scene_intermediate");
  Buffer mat = make_buffer(a, 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "material");
  float diffuse[4] = {0.8f, 0.5f, 0.2f, 1.0f};          // SoMaterial diffuseColor
  memcpy(mat.map, diffuse, sizeof diffuse);

  auto matl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto outl = make_set_layout(a, {{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto mat_set = alloc_set(a, matl, "material_set");
  write_buffer(a, mat_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, mat);
  auto out_set = alloc_set(a, outl, "output_set");
  write_image(a, out_set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, scene.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  GfxDesc vd; vd.vs = vd.fs = sh; vd.vs_entry = "vsVisual"; vd.fs_entry = "fsVisual"; vd.sets = {matl};
  vd.colors = {VK_FORMAT_R16G16B16A16_SFLOAT};
  Pipeline visual = make_graphics(a, vd, "visual");
  GfxDesc od; od.vs = od.fs = sh; od.vs_entry = "vsFullscreen"; od.fs_entry = "fsOutput"; od.sets = {outl};
  od.colors = {a.swap_format}; od.push_size = sizeof(OutputParams);
  Pipeline output = make_graphics(a, od, "output");
  OutputParams params{1u, 1.0f, 203.0f, 0};               // HDR10 swapchain active

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    image_barrier(c, scene.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {scene.view});                  // background: cleared
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, visual.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, visual.layout, 0, 1, &mat_set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    image_barrier(c, scene.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, output.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, output.layout, 0, 1, &out_set, 0, nullptr);
    vkCmdPushConstants(c, output.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof params, &params);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
