// A small engine layer shared by the GPUBugBench reproductions.
//
// Plain Vulkan C API, headless (VK_EXT_headless_surface), so the same
// program runs on a real GPU (Mesa, NVIDIA, AMD all ship the extension) and
// on the COVENANT model driver. Each reproduction keeps only the code that
// matters for its bug; this header holds the boilerplate an engine would.
//
// Every object gets a debug name; the checker reports bugs by these names.
#pragma once
#include <vulkan/vulkan.h>
#include <covenant.h>

#include <cstdio>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define VK(x) do { VkResult r_ = (x); if (r_ < 0) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

namespace bench {

struct Buffer { VkBuffer buf; VkDeviceMemory mem; void* map; VkDeviceSize size; };
struct Image { VkImage img; VkDeviceMemory mem; VkImageView view; VkFormat format; uint32_t w, h, mips; };

struct AppDesc {
  VkFormat swap_format = VK_FORMAT_B8G8R8A8_SRGB;
  uint32_t min_images = 3;
  uint32_t width = 640, height = 480;
};

struct App {
  VkInstance inst{};
  VkPhysicalDevice phys{};
  VkDevice dev{};
  uint32_t gfx_family = 0, comp_family = 0;
  VkQueue gfx{}, comp{};
  VkSurfaceKHR surface{};
  VkSwapchainKHR swapchain{};
  VkFormat swap_format{};
  VkExtent2D extent{};
  std::vector<VkImage> swap_images;
  std::vector<VkImageView> swap_views;
  VkCommandPool gfx_pool{}, comp_pool{};
  VkDescriptorPool desc_pool{};
  VkSampler sampler{};
  std::string shader_dir = ".";
  PFN_vkSetDebugUtilsObjectNameEXT set_name = nullptr;
  VkPhysicalDeviceMemoryProperties memprops{};

  template <class H> void name(H h, VkObjectType t, const std::string& n) {
    if (!set_name) return;
    VkDebugUtilsObjectNameInfoEXT i{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    i.objectType = t; i.objectHandle = (uint64_t)h; i.pObjectName = n.c_str();
    set_name(dev, &i);
  }
};

inline uint32_t mem_type(App& a, uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < a.memprops.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (a.memprops.memoryTypes[i].propertyFlags & want) == want) return i;
  fprintf(stderr, "no memory type\n"); exit(1);
}

inline App make_app(int argc, char** argv, AppDesc d = {}) {
  App a;
  if (argc > 1) a.shader_dir = argv[1];
  const char* iext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
                        VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  ai.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &ai; ici.enabledExtensionCount = 3; ici.ppEnabledExtensionNames = iext;
  VK(vkCreateInstance(&ici, nullptr, &a.inst));
  a.set_name = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(a.inst, "vkSetDebugUtilsObjectNameEXT");

  uint32_t n = 1;
  VK(vkEnumeratePhysicalDevices(a.inst, &n, &a.phys));
  vkGetPhysicalDeviceMemoryProperties(a.phys, &a.memprops);
  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(a.phys, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qp(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(a.phys, &qn, qp.data());
  a.comp_family = UINT32_MAX;
  for (uint32_t i = 0; i < qn; i++) {
    if (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { a.gfx_family = i; break; }
  }
  for (uint32_t i = 0; i < qn; i++)
    if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) a.comp_family = i;
  if (a.comp_family == UINT32_MAX) a.comp_family = a.gfx_family;

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci[2]{};
  uint32_t nq = 0;
  qci[nq] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci[nq].queueFamilyIndex = a.gfx_family;
  qci[nq].queueCount = 1; qci[nq].pQueuePriorities = &prio; nq++;
  if (a.comp_family != a.gfx_family) {
    qci[nq] = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci[nq].queueFamilyIndex = a.comp_family;
    qci[nq].queueCount = 1; qci[nq].pQueuePriorities = &prio; nq++;
  }
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE;
  const char* dext[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &f13; dci.queueCreateInfoCount = nq; dci.pQueueCreateInfos = qci;
  dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = dext;
  VK(vkCreateDevice(a.phys, &dci, nullptr, &a.dev));
  vkGetDeviceQueue(a.dev, a.gfx_family, 0, &a.gfx);
  vkGetDeviceQueue(a.dev, a.comp_family, 0, &a.comp);
  a.name(a.gfx, VK_OBJECT_TYPE_QUEUE, "graphics");
  if (a.comp != a.gfx) a.name(a.comp, VK_OBJECT_TYPE_QUEUE, "compute");

  VkHeadlessSurfaceCreateInfoEXT hs{VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT};
  auto mkSurf = (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(a.inst, "vkCreateHeadlessSurfaceEXT");
  VK(mkSurf(a.inst, &hs, nullptr, &a.surface));
  a.extent = {d.width, d.height};
  a.swap_format = d.swap_format;
  VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sci.surface = a.surface; sci.minImageCount = d.min_images; sci.imageFormat = d.swap_format;
  sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR; sci.imageExtent = a.extent;
  sci.imageArrayLayers = 1; sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  sci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sci.clipped = VK_TRUE;
  VK(vkCreateSwapchainKHR(a.dev, &sci, nullptr, &a.swapchain));
  a.name(a.swapchain, VK_OBJECT_TYPE_SWAPCHAIN_KHR, "swapchain");
  uint32_t ni = 0;
  vkGetSwapchainImagesKHR(a.dev, a.swapchain, &ni, nullptr);
  a.swap_images.resize(ni);
  vkGetSwapchainImagesKHR(a.dev, a.swapchain, &ni, a.swap_images.data());
  for (uint32_t i = 0; i < ni; i++) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = a.swap_images[i]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = d.swap_format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v; VK(vkCreateImageView(a.dev, &vi, nullptr, &v));
    a.swap_views.push_back(v);
  }

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pci.queueFamilyIndex = a.gfx_family;
  VK(vkCreateCommandPool(a.dev, &pci, nullptr, &a.gfx_pool));
  pci.queueFamilyIndex = a.comp_family;
  VK(vkCreateCommandPool(a.dev, &pci, nullptr, &a.comp_pool));

  VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64},
                               {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64}};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 64; dpi.poolSizeCount = 4; dpi.pPoolSizes = ps;
  VK(vkCreateDescriptorPool(a.dev, &dpi, nullptr, &a.desc_pool));
  VkSamplerCreateInfo smp{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  smp.magFilter = smp.minFilter = VK_FILTER_LINEAR; smp.maxLod = 16;
  VK(vkCreateSampler(a.dev, &smp, nullptr, &a.sampler));
  return a;
}

// ------------------------------------------------------------- resources
inline Buffer make_buffer(App& a, VkDeviceSize size, VkBufferUsageFlags usage, bool host, const std::string& nm) {
  Buffer b{}; b.size = size;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size; bi.usage = usage;
  VK(vkCreateBuffer(a.dev, &bi, nullptr, &b.buf));
  VkMemoryRequirements r; vkGetBufferMemoryRequirements(a.dev, b.buf, &r);
  VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mi.allocationSize = r.size;
  mi.memoryTypeIndex = mem_type(a, r.memoryTypeBits, host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                          : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK(vkAllocateMemory(a.dev, &mi, nullptr, &b.mem));
  VK(vkBindBufferMemory(a.dev, b.buf, b.mem, 0));
  if (host) VK(vkMapMemory(a.dev, b.mem, 0, size, 0, &b.map));
  a.name(b.buf, VK_OBJECT_TYPE_BUFFER, nm);
  return b;
}

inline Image make_image(App& a, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, const std::string& nm,
                        uint32_t mips = 1, VkFormat view_fmt = VK_FORMAT_UNDEFINED) {
  Image im{}; im.format = fmt; im.w = w; im.h = h; im.mips = mips;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {w, h, 1}; ii.mipLevels = mips; ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage;
  if (view_fmt != VK_FORMAT_UNDEFINED && view_fmt != fmt) ii.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  VK(vkCreateImage(a.dev, &ii, nullptr, &im.img));
  VkMemoryRequirements r; vkGetImageMemoryRequirements(a.dev, im.img, &r);
  VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mi.allocationSize = r.size; mi.memoryTypeIndex = mem_type(a, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK(vkAllocateMemory(a.dev, &mi, nullptr, &im.mem));
  VK(vkBindImageMemory(a.dev, im.img, im.mem, 0));
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = im.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = view_fmt != VK_FORMAT_UNDEFINED ? view_fmt : fmt;
  bool depth = fmt == VK_FORMAT_D32_SFLOAT || fmt == VK_FORMAT_D24_UNORM_S8_UINT;
  vi.subresourceRange = {(VkImageAspectFlags)(depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, mips, 0, 1};
  VK(vkCreateImageView(a.dev, &vi, nullptr, &im.view));
  a.name(im.img, VK_OBJECT_TYPE_IMAGE, nm);
  a.name(im.view, VK_OBJECT_TYPE_IMAGE_VIEW, nm);
  return im;
}

inline VkShaderModule load_shader(App& a, const std::string& file) {
  std::string path = a.shader_dir + "/" + file;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint32_t> code((n + 3) / 4);
  fread(code.data(), 1, n, f); fclose(f);
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = n; ci.pCode = code.data();
  VkShaderModule m; VK(vkCreateShaderModule(a.dev, &ci, nullptr, &m));
  a.name(m, VK_OBJECT_TYPE_SHADER_MODULE, file);
  return m;
}

// ----------------------------------------------------------- descriptors
struct Binding { uint32_t binding; VkDescriptorType type; VkShaderStageFlags stages; uint32_t count = 1; };

inline VkDescriptorSetLayout make_set_layout(App& a, std::vector<Binding> bs) {
  std::vector<VkDescriptorSetLayoutBinding> v;
  for (auto& b : bs) v.push_back({b.binding, b.type, b.count, b.stages, nullptr});
  VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  ci.bindingCount = (uint32_t)v.size(); ci.pBindings = v.data();
  VkDescriptorSetLayout l; VK(vkCreateDescriptorSetLayout(a.dev, &ci, nullptr, &l));
  return l;
}

inline VkDescriptorSet alloc_set(App& a, VkDescriptorSetLayout l, const std::string& nm) {
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = a.desc_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &l;
  VkDescriptorSet s; VK(vkAllocateDescriptorSets(a.dev, &ai, &s));
  a.name(s, VK_OBJECT_TYPE_DESCRIPTOR_SET, nm);
  return s;
}

inline void write_buffer(App& a, VkDescriptorSet s, uint32_t binding, VkDescriptorType t, const Buffer& b) {
  VkDescriptorBufferInfo bi{b.buf, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet = s; w.dstBinding = binding; w.descriptorCount = 1; w.descriptorType = t; w.pBufferInfo = &bi;
  vkUpdateDescriptorSets(a.dev, 1, &w, 0, nullptr);
}

inline void write_image(App& a, VkDescriptorSet s, uint32_t binding, VkDescriptorType t, VkImageView v,
                        VkImageLayout layout) {
  VkDescriptorImageInfo ii{a.sampler, v, layout};
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet = s; w.dstBinding = binding; w.descriptorCount = 1; w.descriptorType = t; w.pImageInfo = &ii;
  vkUpdateDescriptorSets(a.dev, 1, &w, 0, nullptr);
}

// ------------------------------------------------------------- pipelines
struct GfxDesc {
  VkShaderModule vs{}, fs{};
  const char* vs_entry = "vsMain";
  const char* fs_entry = "fsMain";
  std::vector<VkDescriptorSetLayout> sets;
  std::vector<VkFormat> colors;
  VkFormat depth = VK_FORMAT_UNDEFINED;
  VkCompareOp depth_compare = VK_COMPARE_OP_LESS;
  uint32_t vertex_stride = 0;                                   // 0 = no vertex buffer
  std::vector<VkVertexInputAttributeDescription> attributes;
  uint32_t push_size = 0;
};

struct Pipeline { VkPipeline pipe; VkPipelineLayout layout; };

inline Pipeline make_graphics(App& a, const GfxDesc& d, const std::string& nm) {
  Pipeline p{};
  VkPushConstantRange pr{VK_SHADER_STAGE_ALL_GRAPHICS, 0, d.push_size};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = (uint32_t)d.sets.size(); li.pSetLayouts = d.sets.data();
  li.pushConstantRangeCount = d.push_size ? 1 : 0; li.pPushConstantRanges = &pr;
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  st[0].module = d.vs; st[0].pName = d.vs_entry;
  st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  st[1].module = d.fs; st[1].pName = d.fs_entry;
  VkVertexInputBindingDescription vb{0, d.vertex_stride, VK_VERTEX_INPUT_RATE_VERTEX};
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  if (d.vertex_stride) {
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb;
    vi.vertexAttributeDescriptionCount = (uint32_t)d.attributes.size(); vi.pVertexAttributeDescriptions = d.attributes.data();
  }
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.lineWidth = 1; rs.cullMode = VK_CULL_MODE_NONE;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = ds.depthWriteEnable = d.depth != VK_FORMAT_UNDEFINED; ds.depthCompareOp = d.depth_compare;
  std::vector<VkPipelineColorBlendAttachmentState> cb(d.colors.size());
  for (auto& c : cb) c.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = (uint32_t)cb.size(); cbs.pAttachments = cb.data();
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo rinfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rinfo.colorAttachmentCount = (uint32_t)d.colors.size(); rinfo.pColorAttachmentFormats = d.colors.data();
  rinfo.depthAttachmentFormat = d.depth;
  VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gi.pNext = &rinfo; gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
  gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pDepthStencilState = &ds;
  gi.pColorBlendState = &cbs; gi.pDynamicState = &dy; gi.layout = p.layout;
  VK(vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p.pipe));
  a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, nm);
  return p;
}

inline Pipeline make_compute(App& a, VkShaderModule cs, const char* entry, std::vector<VkDescriptorSetLayout> sets,
                             const std::string& nm) {
  Pipeline p{};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = (uint32_t)sets.size(); li.pSetLayouts = sets.data();
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = cs; ci.stage.pName = entry;
  ci.layout = p.layout;
  VK(vkCreateComputePipelines(a.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipe));
  a.name(p.pipe, VK_OBJECT_TYPE_PIPELINE, nm);
  return p;
}

// ------------------------------------------------------------- commands
inline VkCommandBuffer alloc_cmd(App& a, VkCommandPool pool) {
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
  VkCommandBuffer c; VK(vkAllocateCommandBuffers(a.dev, &ai, &c));
  return c;
}

inline void begin(VkCommandBuffer c) {
  VK(vkResetCommandBuffer(c, 0));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK(vkBeginCommandBuffer(c, &bi));
}

// One image barrier (synchronization2). Masks are given explicitly on purpose:
// wrong masks are a bug class the benchmark covers.
inline void image_barrier(VkCommandBuffer c, VkImage img, VkImageLayout from, VkImageLayout to,
                          VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                          uint32_t base_mip = 0, uint32_t mips = VK_REMAINING_MIP_LEVELS) {
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.srcStageMask = src_stage; b.srcAccessMask = src_access; b.dstStageMask = dst_stage; b.dstAccessMask = dst_access;
  b.oldLayout = from; b.newLayout = to; b.image = img;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bool depth = false;
  b.subresourceRange = {(VkImageAspectFlags)(depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), base_mip, mips, 0, 1};
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
  vkCmdPipelineBarrier2(c, &di);
}

inline void depth_barrier(VkCommandBuffer c, VkImage img, VkImageLayout from, VkImageLayout to,
                          VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.srcStageMask = src_stage; b.srcAccessMask = src_access; b.dstStageMask = dst_stage; b.dstAccessMask = dst_access;
  b.oldLayout = from; b.newLayout = to; b.image = img;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
  vkCmdPipelineBarrier2(c, &di);
}

inline void buffer_barrier(VkCommandBuffer c, VkBuffer buf, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                           VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
  VkBufferMemoryBarrier2 b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  b.srcStageMask = src_stage; b.srcAccessMask = src_access; b.dstStageMask = dst_stage; b.dstAccessMask = dst_access;
  b.buffer = buf; b.size = VK_WHOLE_SIZE;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.bufferMemoryBarrierCount = 1; di.pBufferMemoryBarriers = &b;
  vkCmdPipelineBarrier2(c, &di);
}

inline void begin_rendering(App& a, VkCommandBuffer c, std::vector<VkImageView> colors, VkImageView depth = VK_NULL_HANDLE,
                            float depth_clear = 1.0f) {
  std::vector<VkRenderingAttachmentInfo> ca;
  for (auto v : colors) {
    VkRenderingAttachmentInfo r{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    r.imageView = v; r.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    r.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; r.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ca.push_back(r);
  }
  VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  da.imageView = depth; da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  da.clearValue.depthStencil = {depth_clear, 0};
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, a.extent}; ri.layerCount = 1;
  ri.colorAttachmentCount = (uint32_t)ca.size(); ri.pColorAttachments = ca.data();
  if (depth) ri.pDepthAttachment = &da;
  vkCmdBeginRendering(c, &ri);
  VkViewport vp{0, 0, (float)a.extent.width, (float)a.extent.height, 0, 1};
  VkRect2D sc{{0, 0}, a.extent};
  vkCmdSetViewport(c, 0, 1, &vp);
  vkCmdSetScissor(c, 0, 1, &sc);
}

inline VkSemaphore make_semaphore(App& a, const std::string& nm) {
  VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkSemaphore s; VK(vkCreateSemaphore(a.dev, &si, nullptr, &s));
  a.name(s, VK_OBJECT_TYPE_SEMAPHORE, nm);
  return s;
}

inline VkFence make_fence(App& a, bool signaled, const std::string& nm) {
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (signaled) fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkFence f; VK(vkCreateFence(a.dev, &fi, nullptr, &f));
  a.name(f, VK_OBJECT_TYPE_FENCE, nm);
  return f;
}

// Submit one command buffer: waits (semaphore + stage), signals, fence.
inline void submit(VkQueue q, VkCommandBuffer c, std::vector<std::pair<VkSemaphore, VkPipelineStageFlags>> waits,
                   std::vector<VkSemaphore> signals, VkFence fence) {
  std::vector<VkSemaphore> ws; std::vector<VkPipelineStageFlags> st;
  for (auto& w : waits) { ws.push_back(w.first); st.push_back(w.second); }
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.waitSemaphoreCount = (uint32_t)ws.size(); si.pWaitSemaphores = ws.data(); si.pWaitDstStageMask = st.data();
  si.commandBufferCount = 1; si.pCommandBuffers = &c;
  si.signalSemaphoreCount = (uint32_t)signals.size(); si.pSignalSemaphores = signals.data();
  VK(vkQueueSubmit(q, 1, &si, fence));
}

inline void present(App& a, VkQueue q, uint32_t index, VkSemaphore wait) {
  VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  pi.waitSemaphoreCount = wait ? 1 : 0; pi.pWaitSemaphores = &wait;
  pi.swapchainCount = 1; pi.pSwapchains = &a.swapchain; pi.pImageIndices = &index;
  VK(vkQueuePresentKHR(q, &pi));
}

inline int frames_from_env(int dflt = 6) {
  const char* e = getenv("BENCH_FRAMES");
  return e ? atoi(e) : dflt;
}

// A correct frame loop: per-frame fence, acquire/render-done semaphores,
// swapchain image transitions. `record` fills the frame's command buffer
// after the swapchain image is moved to COLOR_ATTACHMENT (offscreen passes may
// come first in the same buffer); the move to PRESENT_SRC follows.
// `before_submit` runs on the CPU after the fence wait (uploads go there).
inline void run_frames(App& a, int frames, int in_flight,
                       std::function<void(VkCommandBuffer, uint32_t image, int frame)> record,
                       std::function<void(int frame)> before_submit = nullptr) {
  std::vector<VkCommandBuffer> cmd(in_flight);
  // the semaphore a present waits on is in use until that image is acquired
  // again, so render_done is per swapchain image, not per frame in flight
  std::vector<VkSemaphore> acquired(in_flight), done(a.swap_images.size());
  std::vector<VkFence> fence(in_flight);
  for (int i = 0; i < in_flight; i++) {
    cmd[i] = alloc_cmd(a, a.gfx_pool);
    acquired[i] = make_semaphore(a, "acquired[" + std::to_string(i) + "]");
    fence[i] = make_fence(a, true, "frame_fence[" + std::to_string(i) + "]");
  }
  for (size_t i = 0; i < done.size(); i++)
    done[i] = make_semaphore(a, "render_done[" + std::to_string(i) + "]");
  for (int f = 0; f < frames; f++) {
    int k = f % in_flight;
    VK(vkWaitForFences(a.dev, 1, &fence[k], VK_TRUE, UINT64_MAX));
    VK(vkResetFences(a.dev, 1, &fence[k]));
    uint32_t idx;
    VK(vkAcquireNextImageKHR(a.dev, a.swapchain, UINT64_MAX, acquired[k], VK_NULL_HANDLE, &idx));
    if (before_submit) before_submit(f);
    begin(cmd[k]);
    image_barrier(cmd[k], a.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    record(cmd[k], idx, f);
    image_barrier(cmd[k], a.swap_images[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    VK(vkEndCommandBuffer(cmd[k]));
    submit(a.gfx, cmd[k], {{acquired[k], VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}}, {done[idx]}, fence[k]);
    present(a, a.gfx, idx, done[idx]);
  }
  vkDeviceWaitIdle(a.dev);
}

}  // namespace bench
