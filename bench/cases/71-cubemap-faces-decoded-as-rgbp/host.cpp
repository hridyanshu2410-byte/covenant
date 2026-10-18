// Distilled from playcanvas/engine PR #9607 (fixes #9542): a cubemap asset
// whose data.textures lists face image URLs (rather than texture asset ids)
// loaded every face with the texture state meant for a prefiltered env atlas:
// type 'rgbp' (linear, unsigned storage, clamped, no mipmaps). The assembled
// cubemap reported type 'rgbp', so the skybox decoded the plain 8-bit sRGB
// face images as RGBP. The fix creates URL faces as ordinary colour
// textures: srgb = !type || type === TEXTURETYPE_DEFAULT, and the cubemap
// defaults to type 'default' (as with texture asset ids); explicit
// rgbm/rgbp/rgbe types keep their linear data. Port: WebGL -> Vulkan; an
// sRGB texture is an R8G8B8A8_SRGB image (sampled as linear, encoding
// 'linear'), the env-atlas state an R8G8B8A8_UNORM image with encoding 'rgbp'.
#include "../../common/app.hpp"
using namespace bench;

enum TextureType { TEXTURETYPE_DEFAULT, TEXTURETYPE_RGBP };
struct TextureOptions { TextureType type; bool srgb; bool mipmaps; };

// CubemapHandler: options for the faces of cubemapAsset
static TextureOptions face_options_from_urls(TextureType assetDataType, bool hasAssetType) {
#ifdef FIXED
  (void)hasAssetType;
  TextureType type = assetDataType;                         // getAssetDataType(cubemapAsset.data || {})
  return {type, type == TEXTURETYPE_DEFAULT, true};         // data = { srgb: !type || type === TEXTURETYPE_DEFAULT }
#else
  (void)assetDataType; (void)hasAssetType;
  return {TEXTURETYPE_RGBP, false, false};                  // prefiltered env-atlas state
#endif
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);                             // back buffer B8G8R8A8_SRGB
  auto sh = load_shader(a, "shaders.spv");
  // cubemap asset: data = { textures: [6 face URLs] }, no type given
  TextureOptions opt = face_options_from_urls(TEXTURETYPE_DEFAULT, false);
  VkFormat fmt = opt.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
  uint32_t encoding = opt.type == TEXTURETYPE_RGBP ? 2u : (opt.srgb ? 0u : 1u);
  const uint32_t S = 64;

  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt;
  ii.extent = {S, S, 1}; ii.mipLevels = 1; ii.arrayLayers = 6; ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  VkImage cube; VK(vkCreateImage(a.dev, &ii, nullptr, &cube));
  VkMemoryRequirements r; vkGetImageMemoryRequirements(a.dev, cube, &r);
  VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mi.allocationSize = r.size; mi.memoryTypeIndex = mem_type(a, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkDeviceMemory mem; VK(vkAllocateMemory(a.dev, &mi, nullptr, &mem)); VK(vkBindImageMemory(a.dev, cube, mem, 0));
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = cube; vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE; vi.format = fmt; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
  VkImageView cubeView; VK(vkCreateImageView(a.dev, &vi, nullptr, &cubeView));
  a.name(cube, VK_OBJECT_TYPE_IMAGE, "skybox_cubemap"); a.name(cubeView, VK_OBJECT_TYPE_IMAGE_VIEW, "skybox_cubemap");

  // the six face PNGs (8-bit sRGB, opaque)
  Buffer faces = make_buffer(a, 6 * S * S * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "cubemap_face_images");
  uint8_t* p = (uint8_t*)faces.map;
  for (uint32_t i = 0; i < 6 * S * S; i++) { p[i * 4] = 110; p[i * 4 + 1] = 160; p[i * 4 + 2] = 220; p[i * 4 + 3] = 255; }
  VkCommandBuffer up = alloc_cmd(a, a.gfx_pool);
  begin(up);
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.srcStageMask = VK_PIPELINE_STAGE_2_NONE; b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.image = cube;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO}; di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
  vkCmdPipelineBarrier2(up, &di);
  VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 6}; region.imageExtent = {S, S, 1};
  vkCmdCopyBufferToImage(up, faces.buf, cube, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT; b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  vkCmdPipelineBarrier2(up, &di);
  VK(vkEndCommandBuffer(up));
  submit(a.gfx, up, {}, {}, VK_NULL_HANDLE);
  VK(vkQueueWaitIdle(a.gfx));

  Buffer ubo = make_buffer(a, 80, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "skybox_uniforms");
  struct { float m[16]; uint32_t encoding; float exposure; float pad[2]; } u = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, encoding, 1.0f, {0, 0}};
  memcpy(ubo.map, &u, sizeof u);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set = alloc_set(a, setl, "skybox_material");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_image(a, set, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, cubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vsSkybox"; gd.fs_entry = "fsSkybox"; gd.sets = {setl}; gd.colors = {a.swap_format};
  Pipeline skybox = make_graphics(a, gd, "skybox");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, skybox.layout, 0, 1, &set, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
