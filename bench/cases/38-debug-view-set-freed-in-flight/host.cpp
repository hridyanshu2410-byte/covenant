// Distilled from TommyRadan/AlphaEngine PR #278: the debug overlay's
// render-target viewer caches one ImGui Vulkan descriptor set per texture
// handle (HDR, LDR, velocity, TAA). A resize recreates the targets and
// released the old texture's set at once (ImGui_ImplVulkan_RemoveTexture ->
// vkFreeDescriptorSets), bypassing the device's deferred-destroy queue. With
// two frames in flight (PR #270) the previous frame's command buffer could
// still be reading that set (VUID-vkFreeDescriptorSets-pDescriptorSets-00309).
// The fix routes the release through vk_device::enqueue_destroy, gated on the
// submission serial that may still reference the set, and drains the queue
// after a queue idle at shutdown (flush_pending_destroys).
#include "../../common/app.hpp"
#include <map>
using namespace bench;

static const int FRAMES_IN_FLIGHT = 2;

struct Texture { Image image; bool initialized = false; };

struct DeferredDestroy { uint64_t serial; std::function<void()> fn; };

struct VkDeviceWrap {
  App* a;
  uint64_t submitted_serial = 0;            // serial of the last frame submitted
  std::vector<DeferredDestroy> pending;
  void enqueue_destroy(std::function<void()> fn) { pending.push_back({submitted_serial, fn}); }
  void collect(uint64_t completed_serial) {
    for (size_t i = 0; i < pending.size();)
      if (pending[i].serial <= completed_serial) { pending[i].fn(); pending.erase(pending.begin() + i); } else i++;
  }
  void flush_pending_destroys() {
    vkQueueWaitIdle(a->gfx);
    for (auto& d : pending) d.fn();
    pending.clear();
  }
};

struct RenderTargetViewer {
  App* a; VkDeviceWrap* device;
  VkDescriptorPool imgui_pool; VkDescriptorSetLayout layout;
  std::map<Texture*, VkDescriptorSet> bindings;      // one ImGui set per texture handle

  VkDescriptorSet get_binding(Texture* t) {
    auto it = bindings.find(t);
    if (it != bindings.end()) return it->second;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = imgui_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &layout;
    VkDescriptorSet s; VK(vkAllocateDescriptorSets(a->dev, &ai, &s));
    a->name(s, VK_OBJECT_TYPE_DESCRIPTOR_SET, "imgui_texture_set");
    write_image(*a, s, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, t->image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    bindings[t] = s;
    return s;
  }
  // ImGui_ImplVulkan_RemoveTexture
  void release_binding(Texture* t) {
    auto it = bindings.find(t);
    if (it == bindings.end()) return;
    VkDescriptorSet s = it->second;
    bindings.erase(it);
#ifdef FIXED
    defer_release_vulkan_texture_binding(s);
#else
    VK(vkFreeDescriptorSets(a->dev, imgui_pool, 1, &s));
#endif
  }
#ifdef FIXED
  void defer_release_vulkan_texture_binding(VkDescriptorSet s) {
    App* app = a; VkDescriptorPool pool = imgui_pool;
    device->enqueue_destroy([app, pool, s]() mutable { VK(vkFreeDescriptorSets(app->dev, pool, 1, &s)); });
  }
#endif
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  VkDeviceWrap device{&a};
  RenderTargetViewer viewer{&a, &device};
  {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT; dpi.maxSets = 16; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VK(vkCreateDescriptorPool(a.dev, &dpi, nullptr, &viewer.imgui_pool));
  }
  viewer.layout = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {viewer.layout}; gd.colors = {a.swap_format};
  Pipeline imgui = make_graphics(a, gd, "imgui_image");

  auto make_hdr = [&](uint32_t w, uint32_t h) {
    auto* t = new Texture{make_image(a, w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, "hdr_target")};
    return t;
  };
  Texture* hdr = make_hdr(640, 480);
  const int FRAMES = frames_from_env(6), RESIZE_AT = 3;

  run_frames(a, FRAMES, FRAMES_IN_FLIGHT, [&](VkCommandBuffer c, uint32_t idx, int f) {
    if (!hdr->initialized) {
      image_barrier(c, hdr->image.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
      hdr->initialized = true;
    }
    VkDescriptorSet s = viewer.get_binding(hdr);      // the viewer shows the HDR target
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, imgui.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, imgui.layout, 0, 1, &s, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    device.submitted_serial = (uint64_t)f + 1;
  }, [&](int f) {
    // gpu::begin_frame: this slot's fence has signaled, so frame f - FRAMES_IN_FLIGHT is complete
    if (f >= FRAMES_IN_FLIGHT) device.collect((uint64_t)(f - FRAMES_IN_FLIGHT + 1));
    if (f == RESIZE_AT) {                              // window resized: recreate the targets
      Texture* old = hdr;
      hdr = make_hdr(800, 600);
      viewer.release_binding(old);
    }
  });
#ifdef FIXED
  device.flush_pending_destroys();
#endif
  return 0;
}
