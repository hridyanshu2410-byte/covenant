// Distilled from DisplayXR/displayxr-runtime #1792 (open): the Vulkan
// compositor caches one descriptor set per window-space layer image (the
// window-space view cache). vk_compositor_forget_dead_window_space_images()
// collects images that died between passes into a fixed-size list; when more
// images die than the list holds, the overflow path frees the descriptor sets
// of every cached entry and clears the cache. It assumes all earlier
// submissions have completed, which holds on the windowed and shared-texture
// paths (fence / vkQueueWaitIdle), but vk_split_composite_window_space uses a
// three-deep command ring that waits only on the slot it reuses, so up to two
// earlier command buffers that bind the HUD's cached set can still be
// executing: VUID-vkFreeDescriptorSets-pDescriptorSets-00309. The fixed
// variant takes the issue's first proposal: synchronize with the outstanding
// command buffers before evicting on the split path. The dead-list capacity
// is 4 here (64 upstream).
#include "../../common/app.hpp"
#include <map>
using namespace bench;

static const size_t kMaxDeadImages = 4;

struct ViewCache {
  App* a; VkDescriptorPool pool; VkDescriptorSetLayout layout;
  std::map<VkImageView, VkDescriptorSet> entries;
  VkDescriptorSet get(VkImageView v, const std::string& nm) {
    auto it = entries.find(v);
    if (it != entries.end()) return it->second;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &layout;
    VkDescriptorSet s; VK(vkAllocateDescriptorSets(a->dev, &ai, &s));
    a->name(s, VK_OBJECT_TYPE_DESCRIPTOR_SET, "window_space_set:" + nm);
    write_image(*a, s, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    entries[v] = s;
    return s;
  }
  // vk_compositor_forget_dead_window_space_images
  void forget_dead(const std::vector<VkImageView>& dead) {
    if (dead.size() > kMaxDeadImages) {        // overflow: drop the whole cache
      for (auto& e : entries) VK(vkFreeDescriptorSets(a->dev, pool, 1, &e.second));
      entries.clear();
      return;
    }
    for (auto v : dead) {
      auto it = entries.find(v);
      if (it == entries.end()) continue;
      VK(vkFreeDescriptorSets(a->dev, pool, 1, &it->second));
      entries.erase(it);
    }
  }
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT; dpi.maxSets = 32; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
  ViewCache cache{&a};
  VK(vkCreateDescriptorPool(a.dev, &dpi, nullptr, &cache.pool));
  cache.layout = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "vsQuad"; gd.fs_entry = "fsLayer"; gd.sets = {cache.layout};
  gd.colors = {a.swap_format}; gd.push_size = 16;
  Pipeline layer = make_graphics(a, gd, "window_space_layer");

  // layer images, already uploaded and in SHADER_READ_ONLY_OPTIMAL
  auto make_layer_image = [&](const std::string& nm) {
    Image im = make_image(a, 256, 256, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, nm);
    VkCommandBuffer c = alloc_cmd(a, a.gfx_pool); begin(c);
    image_barrier(c, im.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    VK(vkEndCommandBuffer(c)); submit(a.gfx, c, {}, {}, VK_NULL_HANDLE); VK(vkQueueWaitIdle(a.gfx));
    return im;
  };
  Image hud = make_layer_image("hud");
  std::vector<Image> app_images, new_app_images;
  for (int i = 0; i < 5; i++) app_images.push_back(make_layer_image("app_window[" + std::to_string(i) + "]"));
  for (int i = 0; i < 5; i++) new_app_images.push_back(make_layer_image("app_window_v2[" + std::to_string(i) + "]"));

  // vk_split_composite_window_space: three-deep command ring, waits only on the reused slot
  run_frames(a, frames_from_env(6), 3, [&](VkCommandBuffer c, uint32_t idx, int f) {
    std::vector<Image>* app = &app_images;
    if (f >= 3) app = &new_app_images;
    if (f == 3) {      // the app recreated its window-space swapchain: five images died between passes
      std::vector<VkImageView> dead;
      for (auto& im : app_images) dead.push_back(im.view);
#ifdef FIXED
      VK(vkQueueWaitIdle(a.gfx));             // outstanding ring slots complete before eviction
#endif
      cache.forget_dead(dead);
    }
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layer.pipe);
    const Image* layers[2] = {&(*app)[f % 5], &hud};
    const char* names[2] = {"app_window", "hud"};
    for (int l = 0; l < 2; l++) {
      VkDescriptorSet s = cache.get(layers[l]->view, names[l]);
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layer.layout, 0, 1, &s, 0, nullptr);
      float rect[4] = {l ? 0.05f : 0.0f, l ? 0.05f : 0.0f, l ? 0.3f : 1.0f, l ? 0.1f : 1.0f};
      vkCmdPushConstants(c, layer.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof rect, rect);
      vkCmdDraw(c, 6, 1, 0, 0);
    }
    vkCmdEndRendering(c);
  });
}
