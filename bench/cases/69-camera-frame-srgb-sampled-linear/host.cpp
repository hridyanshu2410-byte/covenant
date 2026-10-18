// Distilled from Fabric-Project/Fabric #381 (open; fix proposed in the
// issue): the document renders in linear rgba16Float and is encoded to sRGB
// for display. Camera Provider and Syphon Client wrap the incoming frames as
// .bgra8Unorm textures; the pixels are display-referred sRGB, but nothing
// decodes them, so the linear pipeline treats sRGB code values as linear
// light and the output encode applies the transfer curve a second time
// (paler, brighter colours; inserting an "sRGB to RGB Linear" node fixes it).
// The issue's first proposed fix: sample the frame through an sRGB view,
// makeTextureView(pixelFormat: .bgra8Unorm_srgb). Port: Metal -> Vulkan; the
// CVPixelBuffer -> MTLTexture wrap becomes a staging copy into a
// B8G8R8A8_UNORM image (MUTABLE_FORMAT so an sRGB view is allowed).
#include "../../common/app.hpp"
using namespace bench;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);                    // drawable: B8G8R8A8_SRGB
  auto sh = load_shader(a, "shaders.spv");
  const uint32_t CW = 320, CH = 240;
#ifdef FIXED
  VkFormat camera_view_format = VK_FORMAT_B8G8R8A8_SRGB;     // makeTextureView(pixelFormat: .bgra8Unorm_srgb)
#else
  VkFormat camera_view_format = VK_FORMAT_B8G8R8A8_UNORM;    // sampled as .bgra8Unorm
#endif
  Image camera = make_image(a, CW, CH, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            "camera_provider_texture", 1, VK_FORMAT_B8G8R8A8_SRGB);
  // the texture view the Image Mesh samples
  vkDestroyImageView(a.dev, camera.view, nullptr);
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = camera.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = camera_view_format;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VK(vkCreateImageView(a.dev, &vi, nullptr, &camera.view));
  a.name(camera.view, VK_OBJECT_TYPE_IMAGE_VIEW, "camera_provider_texture");
  Buffer frame = make_buffer(a, CW * CH * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "CVPixelBuffer");
  Image document = make_image(a, 640, 480, VK_FORMAT_R16G16B16A16_SFLOAT,
                              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, "document_rgba16Float");

  auto setl0 = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto setl1 = make_set_layout(a, {{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto meshSet = alloc_set(a, setl0, "image_mesh_material");
  write_image(a, meshSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, camera.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  auto outSet = alloc_set(a, setl1, "output_pass");
  write_image(a, outSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, document.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc md; md.vs = md.fs = sh; md.vs_entry = "vsQuad"; md.fs_entry = "fsImageMesh"; md.sets = {setl0};
  md.colors = {VK_FORMAT_R16G16B16A16_SFLOAT};
  Pipeline imageMesh = make_graphics(a, md, "image_mesh");
  GfxDesc od; od.vs = od.fs = sh; od.vs_entry = "vsFullscreen"; od.fs_entry = "fsOutput"; od.sets = {setl1};
  od.colors = {a.swap_format};
  Pipeline output = make_graphics(a, od, "document_output");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    // new camera frame -> texture
    image_barrier(c, camera.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; region.imageExtent = {CW, CH, 1};
    vkCmdCopyBufferToImage(c, frame.buf, camera.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    image_barrier(c, camera.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    // document render (linear)
    image_barrier(c, document.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {document.view});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, imageMesh.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, imageMesh.layout, 0, 1, &meshSet, 0, nullptr);
    vkCmdDraw(c, 6, 1, 0, 0);
    vkCmdEndRendering(c);
    image_barrier(c, document.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    // output to the sRGB drawable
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, output.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, output.layout, 0, 1, &outSet, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  }, [&](int f) {
    // CVPixelBuffer from the camera: display-referred sRGB BGRA
    uint8_t* p = (uint8_t*)frame.map;
    for (uint32_t i = 0; i < CW * CH; i++) { p[i * 4 + 0] = 40; p[i * 4 + 1] = 60; p[i * 4 + 2] = 220; p[i * 4 + 3] = 255; }
  });
}
