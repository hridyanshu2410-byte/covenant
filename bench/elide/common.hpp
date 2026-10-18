// Shared scaffolding for the COVENANT barrier-elision demonstration workloads.
//
// Each workload is a headless, offscreen Vulkan 1.3 program (no window, no
// swapchain) that is deliberately over-synchronized the way real engines are.
// It runs unchanged on:
//   * the COVENANT model driver (libcovvk, no GPU) to record a trace,
//   * Mesa lavapipe (CPU Vulkan) for correctness checks,
//   * a real GPU for timing.
//
// Command line:  <workload> [--frames N] [--scale S] [--json out.json] [--quiet]
// Environment:
//   ELIDE_VALIDATION=1   enable VK_LAYER_KHRONOS_validation with synchronization
//                        validation; every message is counted and printed.
//   ELIDE_DEVICE=<text>  pick the physical device whose name contains <text>
//                        (default: first discrete GPU, else device 0).
//
// Output: one JSON object (stdout and --json file) with the readback hash of
// every output, per-frame GPU time from timestamp queries and per-frame CPU
// submit-to-fence time.
//
// The command stream must not depend on the device: the plan produced from
// the model-driver trace is matched by command-buffer recording order and
// barrier order, so a workload records exactly the same commands everywhere
// (only image sizes / iteration counts scale with --scale, never the number
// or order of commands).
#pragma once
#include <vulkan/vulkan.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <algorithm>

#define VK(x) do { VkResult r_ = (x); if (r_ < 0) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)r_); exit(1); } } while (0)

namespace el {

struct Buffer { VkBuffer buf{}; VkDeviceMemory mem{}; void* map = nullptr; VkDeviceSize size = 0; };
struct Image { VkImage img{}; VkDeviceMemory mem{}; VkImageView view{}; VkFormat format{}; uint32_t w = 0, h = 0; };

struct Args {
  int frames = 3;
  float scale = 1.0f;
  std::string json;
  bool quiet = false;
};

inline Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--frames" && i + 1 < argc) a.frames = atoi(argv[++i]);
    else if (s == "--scale" && i + 1 < argc) a.scale = (float)atof(argv[++i]);
    else if (s == "--json" && i + 1 < argc) a.json = argv[++i];
    else if (s == "--quiet") a.quiet = true;
    else { fprintf(stderr, "unknown argument %s\n", s.c_str()); exit(2); }
  }
  if (a.frames < 1) a.frames = 1;
  return a;
}

static int g_validation_errors = 0;
static int g_validation_sync_hazards = 0;

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT sev,
                                               VkDebugUtilsMessageTypeFlagsEXT,
                                               const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
  if (sev & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)) {
    g_validation_errors++;
    const char* id = d->pMessageIdName ? d->pMessageIdName : "";
    if (strstr(id, "SYNC-")) g_validation_sync_hazards++;
    fprintf(stderr, "[validation] %s: %s\n", id, d->pMessage ? d->pMessage : "");
  }
  return VK_FALSE;
}

struct Ctx {
  Args args;
  const char* workload = "";
  VkInstance inst{};
  VkPhysicalDevice phys{};
  VkDevice dev{};
  VkQueue queue{};
  uint32_t family = 0;
  VkCommandPool pool{};
  VkCommandBuffer cb{};
  VkFence fence{};
  VkQueryPool qp{};
  bool timestamps = false;
  double ts_period_ns = 1.0;
  uint64_t ts_mask = ~0ull;
  VkDescriptorPool dpool{};
  VkPhysicalDeviceMemoryProperties mp{};
  PFN_vkSetDebugUtilsObjectNameEXT set_name = nullptr;
  VkDebugUtilsMessengerEXT messenger{};
  std::string device_name;
  std::vector<double> gpu_ms, cpu_ms;
  std::vector<std::pair<std::string, std::string>> hashes;  // (output name, hex)

  template <class H> void name(H h, VkObjectType t, const std::string& n) {
    if (!set_name || !h) return;
    VkDebugUtilsObjectNameInfoEXT i{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    i.objectType = t; i.objectHandle = (uint64_t)h; i.pObjectName = n.c_str();
    set_name(dev, &i);
  }
};

inline bool has_ext(const std::vector<VkExtensionProperties>& v, const char* n) {
  for (auto& e : v) if (!strcmp(e.extensionName, n)) return true;
  return false;
}

inline Ctx init(int argc, char** argv, const char* workload) {
  Ctx c;
  c.args = parse_args(argc, argv);
  c.workload = workload;
  bool validation = getenv("ELIDE_VALIDATION") && atoi(getenv("ELIDE_VALIDATION")) != 0;

  uint32_t n = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> exts(n);
  vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data());
  std::vector<const char*> iext, layers;
  bool debug_utils = has_ext(exts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  if (debug_utils) iext.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  VkValidationFeatureEnableEXT vf_en[] = {VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};
  VkValidationFeaturesEXT vf{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  vf.enabledValidationFeatureCount = 1; vf.pEnabledValidationFeatures = vf_en;
  const void* inst_next = nullptr;
  if (validation) {
    layers.push_back("VK_LAYER_KHRONOS_validation");
    uint32_t ln = 0;
    vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &ln, nullptr);
    std::vector<VkExtensionProperties> lext(ln);
    vkEnumerateInstanceExtensionProperties("VK_LAYER_KHRONOS_validation", &ln, lext.data());
    if (has_ext(lext, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME)) {
      iext.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
      inst_next = &vf;
    }
    if (!debug_utils && has_ext(lext, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
      iext.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
      debug_utils = true;
    }
  }
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  ai.pApplicationName = workload;
  ai.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pNext = inst_next;
  ici.pApplicationInfo = &ai;
  ici.enabledExtensionCount = (uint32_t)iext.size(); ici.ppEnabledExtensionNames = iext.data();
  ici.enabledLayerCount = (uint32_t)layers.size(); ici.ppEnabledLayerNames = layers.data();
  VK(vkCreateInstance(&ici, nullptr, &c.inst));
  if (debug_utils) {
    c.set_name = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(c.inst, "vkSetDebugUtilsObjectNameEXT");
    if (validation) {
      auto mk = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(c.inst, "vkCreateDebugUtilsMessengerEXT");
      VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
      mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      mi.pfnUserCallback = debug_cb;
      if (mk) VK(mk(c.inst, &mi, nullptr, &c.messenger));
    }
  }

  // physical device: ELIDE_DEVICE substring, else first discrete GPU, else #0
  n = 0;
  VK(vkEnumeratePhysicalDevices(c.inst, &n, nullptr));
  std::vector<VkPhysicalDevice> pds(n);
  VK(vkEnumeratePhysicalDevices(c.inst, &n, pds.data()));
  if (pds.empty()) { fprintf(stderr, "no Vulkan device\n"); exit(1); }
  const char* want = getenv("ELIDE_DEVICE");
  c.phys = pds[0];
  bool chosen = false;
  for (auto p : pds) {
    VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(p, &pr);
    if (want && strstr(pr.deviceName, want)) { c.phys = p; chosen = true; break; }
  }
  if (!chosen && !want)
    for (auto p : pds) {
      VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(p, &pr);
      if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { c.phys = p; break; }
    }
  VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c.phys, &props);
  c.device_name = props.deviceName;
  c.ts_period_ns = props.limits.timestampPeriod;
  vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mp);

  n = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &n, nullptr);
  std::vector<VkQueueFamilyProperties> qf(n);
  vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &n, qf.data());
  c.family = 0;
  for (uint32_t i = 0; i < n; i++)
    if ((qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { c.family = i; break; }
  uint32_t valid_bits = qf[c.family].timestampValidBits;
  c.timestamps = valid_bits > 0 && c.ts_period_ns > 0;
  c.ts_mask = valid_bits >= 64 ? ~0ull : ((1ull << valid_bits) - 1);

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = c.family; qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.synchronization2 = VK_TRUE; f13.dynamicRendering = VK_TRUE;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &f13;
  dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
  VK(vkCreateDevice(c.phys, &dci, nullptr, &c.dev));
  vkGetDeviceQueue(c.dev, c.family, 0, &c.queue);

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.queueFamilyIndex = c.family; pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VK(vkCreateCommandPool(c.dev, &pci, nullptr, &c.pool));
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = c.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
  VK(vkAllocateCommandBuffers(c.dev, &cai, &c.cb));
  c.name(c.cb, VK_OBJECT_TYPE_COMMAND_BUFFER, "frame_cb");
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VK(vkCreateFence(c.dev, &fci, nullptr, &c.fence));
  VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2;
  VK(vkCreateQueryPool(c.dev, &qpi, nullptr, &c.qp));

  VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
                               {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64}};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 128; dpi.poolSizeCount = 4; dpi.pPoolSizes = ps;
  VK(vkCreateDescriptorPool(c.dev, &dpi, nullptr, &c.dpool));
  return c;
}

inline uint32_t mem_type(Ctx& c, uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < c.mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (c.mp.memoryTypes[i].propertyFlags & want) == want) return i;
  fprintf(stderr, "no memory type\n"); exit(1);
}

inline Buffer make_buffer(Ctx& c, VkDeviceSize size, VkBufferUsageFlags usage, bool host, const std::string& nm) {
  Buffer b; b.size = size;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK(vkCreateBuffer(c.dev, &bi, nullptr, &b.buf));
  c.name(b.buf, VK_OBJECT_TYPE_BUFFER, nm);
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev, b.buf, &mr);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  mai.memoryTypeIndex = mem_type(c, mr.memoryTypeBits, host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                                            : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK(vkAllocateMemory(c.dev, &mai, nullptr, &b.mem));
  VK(vkBindBufferMemory(c.dev, b.buf, b.mem, 0));
  if (host) VK(vkMapMemory(c.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
  return b;
}

inline Image make_image(Ctx& c, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, const std::string& nm) {
  Image im; im.w = w; im.h = h; im.format = fmt;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {w, h, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage;
  ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK(vkCreateImage(c.dev, &ii, nullptr, &im.img));
  c.name(im.img, VK_OBJECT_TYPE_IMAGE, nm);
  VkMemoryRequirements mr; vkGetImageMemoryRequirements(c.dev, im.img, &mr);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size; mai.memoryTypeIndex = mem_type(c, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK(vkAllocateMemory(c.dev, &mai, nullptr, &im.mem));
  VK(vkBindImageMemory(c.dev, im.img, im.mem, 0));
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = im.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VK(vkCreateImageView(c.dev, &vi, nullptr, &im.view));
  c.name(im.view, VK_OBJECT_TYPE_IMAGE_VIEW, nm + "_view");
  return im;
}

inline VkShaderModule shader(Ctx& c, const uint32_t* code, size_t bytes) {
  VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  si.codeSize = bytes; si.pCode = code;
  VkShaderModule m; VK(vkCreateShaderModule(c.dev, &si, nullptr, &m));
  return m;
}

// descriptor set layout from a list of (type) for bindings 0..n-1, all visible to `stages`
inline VkDescriptorSetLayout set_layout(Ctx& c, std::vector<VkDescriptorType> types, VkShaderStageFlags stages) {
  std::vector<VkDescriptorSetLayoutBinding> b(types.size());
  for (size_t i = 0; i < types.size(); i++) b[i] = {(uint32_t)i, types[i], 1, stages, nullptr};
  VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  li.bindingCount = (uint32_t)b.size(); li.pBindings = b.data();
  VkDescriptorSetLayout l; VK(vkCreateDescriptorSetLayout(c.dev, &li, nullptr, &l));
  return l;
}

inline VkPipelineLayout pipe_layout(Ctx& c, VkDescriptorSetLayout sl, uint32_t push_bytes, VkShaderStageFlags push_stages) {
  VkPushConstantRange pr{push_stages, 0, push_bytes};
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = sl ? 1 : 0; pli.pSetLayouts = sl ? &sl : nullptr;
  pli.pushConstantRangeCount = push_bytes ? 1 : 0; pli.pPushConstantRanges = push_bytes ? &pr : nullptr;
  VkPipelineLayout l; VK(vkCreatePipelineLayout(c.dev, &pli, nullptr, &l));
  return l;
}

inline VkDescriptorSet alloc_set(Ctx& c, VkDescriptorSetLayout l, const std::string& nm) {
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = c.dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &l;
  VkDescriptorSet s; VK(vkAllocateDescriptorSets(c.dev, &ai, &s));
  c.name(s, VK_OBJECT_TYPE_DESCRIPTOR_SET, nm);
  return s;
}

// a write per binding; each entry is either a buffer or an image (+ optional sampler)
struct Desc { VkDescriptorType type; VkBuffer buf = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE;
              VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL; VkSampler sampler = VK_NULL_HANDLE; };

inline void write_set(Ctx& c, VkDescriptorSet s, std::vector<Desc> d) {
  std::vector<VkWriteDescriptorSet> w(d.size());
  std::vector<VkDescriptorBufferInfo> bi(d.size());
  std::vector<VkDescriptorImageInfo> ii(d.size());
  for (size_t i = 0; i < d.size(); i++) {
    w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[i].dstSet = s; w[i].dstBinding = (uint32_t)i; w[i].descriptorCount = 1; w[i].descriptorType = d[i].type;
    if (d[i].buf) { bi[i] = {d[i].buf, 0, VK_WHOLE_SIZE}; w[i].pBufferInfo = &bi[i]; }
    else { ii[i] = {d[i].sampler, d[i].view, d[i].layout}; w[i].pImageInfo = &ii[i]; }
  }
  vkUpdateDescriptorSets(c.dev, (uint32_t)w.size(), w.data(), 0, nullptr);
}

inline VkPipeline compute_pipeline(Ctx& c, VkPipelineLayout l, const uint32_t* code, size_t bytes, const std::string& nm) {
  VkShaderModule m = shader(c, code, bytes);
  VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = m; ci.stage.pName = "main";
  ci.layout = l;
  VkPipeline p; VK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p));
  c.name(p, VK_OBJECT_TYPE_PIPELINE, nm);
  return p;
}

// fullscreen-style graphics pipeline for dynamic rendering, no vertex input
inline VkPipeline graphics_pipeline(Ctx& c, VkPipelineLayout l, const uint32_t* vs, size_t vsb, const uint32_t* fs, size_t fsb,
                                    VkFormat color, const std::string& nm, bool blend_add = false) {
  VkShaderModule vm = shader(c, vs, vsb), fm = shader(c, fs, fsb);
  VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = vm; st[0].pName = "main";
  st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fm; st[1].pName = "main";
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState ba{};
  ba.colorWriteMask = 0xF;
  if (blend_add) {
    ba.blendEnable = VK_TRUE;
    ba.srcColorBlendFactor = ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE; ba.colorBlendOp = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE; ba.alphaBlendOp = VK_BLEND_OP_ADD;
  }
  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1; cb.pAttachments = &ba;
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  ds.dynamicStateCount = 2; ds.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  ri.colorAttachmentCount = 1; ri.pColorAttachmentFormats = &color;
  VkGraphicsPipelineCreateInfo gi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gi.pNext = &ri; gi.stageCount = 2; gi.pStages = st; gi.pVertexInputState = &vi; gi.pInputAssemblyState = &ia;
  gi.pViewportState = &vp; gi.pRasterizationState = &rs; gi.pMultisampleState = &ms; gi.pColorBlendState = &cb;
  gi.pDynamicState = &ds; gi.layout = l;
  VkPipeline p; VK(vkCreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gi, nullptr, &p));
  c.name(p, VK_OBJECT_TYPE_PIPELINE, nm);
  return p;
}

inline VkSampler nearest_sampler(Ctx& c) {
  VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  si.magFilter = si.minFilter = VK_FILTER_NEAREST;
  si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkSampler s; VK(vkCreateSampler(c.dev, &si, nullptr, &s));
  return s;
}

// ---------------------------------------------------------------- barriers
// The "engine" barrier helpers. ALL_COMMANDS + MEMORY_READ|WRITE is what
// over-cautious engines emit after every pass.
constexpr VkPipelineStageFlags2 ALL = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
constexpr VkAccessFlags2 RW = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

inline void full_barrier(VkCommandBuffer cb) {
  VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  mb.srcStageMask = ALL; mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  mb.dstStageMask = ALL; mb.dstAccessMask = RW;
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
  vkCmdPipelineBarrier2(cb, &di);
}

inline VkImageMemoryBarrier2 image_barrier(VkImage img, VkImageLayout from, VkImageLayout to,
                                           VkPipelineStageFlags2 ss = ALL, VkAccessFlags2 sa = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                           VkPipelineStageFlags2 ds = ALL, VkAccessFlags2 da = RW) {
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.srcStageMask = ss; b.srcAccessMask = sa; b.dstStageMask = ds; b.dstAccessMask = da;
  b.oldLayout = from; b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  return b;
}

inline void image_barriers(VkCommandBuffer cb, const std::vector<VkImageMemoryBarrier2>& v) {
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.imageMemoryBarrierCount = (uint32_t)v.size(); di.pImageMemoryBarriers = v.data();
  vkCmdPipelineBarrier2(cb, &di);
}

// the classic sync1 spelling, as older engines write it
inline void full_barrier_v1(VkCommandBuffer cb) {
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// ---------------------------------------------------------------- frame loop
inline void begin(Ctx& c) {
  VK(vkResetCommandBuffer(c.cb, 0));
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK(vkBeginCommandBuffer(c.cb, &bi));
}

inline double submit_and_wait(Ctx& c) {
  VK(vkEndCommandBuffer(c.cb));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &c.cb;
  auto t0 = std::chrono::steady_clock::now();
  VK(vkQueueSubmit(c.queue, 1, &si, c.fence));
  VK(vkWaitForFences(c.dev, 1, &c.fence, VK_TRUE, UINT64_MAX));
  auto t1 = std::chrono::steady_clock::now();
  VK(vkResetFences(c.dev, 1, &c.fence));
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// record(cb, frame) records one frame's work between two timestamps.
inline void run_frames(Ctx& c, const std::function<void(VkCommandBuffer, int)>& record) {
  for (int f = 0; f < c.args.frames; f++) {
    begin(c);
    vkCmdResetQueryPool(c.cb, c.qp, 0, 2);
    vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c.qp, 0);
    record(c.cb, f);
    vkCmdWriteTimestamp(c.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c.qp, 1);
    double cpu = submit_and_wait(c);
    c.cpu_ms.push_back(cpu);
    uint64_t ts[2] = {0, 0};
    if (c.timestamps &&
        vkGetQueryPoolResults(c.dev, c.qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
      c.gpu_ms.push_back((double)((ts[1] - ts[0]) & c.ts_mask) * c.ts_period_ns * 1e-6);
  }
}

inline uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  auto* b = (const uint8_t*)p;
  for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}

// Read back images (in `layout`, made available by the caller's last frame)
// and buffers, hash each. One extra command buffer recording.
struct Readback { std::string name; Image* img = nullptr; Buffer* buf = nullptr; VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL; uint32_t texel_bytes = 4; };

inline void readback_and_hash(Ctx& c, std::vector<Readback> outs) {
  // one host buffer per output (the checker does not model buffer ranges)
  std::vector<Buffer> host;
  for (auto& o : outs)
    host.push_back(make_buffer(c, o.img ? (VkDeviceSize)o.img->w * o.img->h * o.texel_bytes : o.buf->size,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, "readback_" + o.name));
  begin(c);
  // the readback pass: everything before -> transfer
  std::vector<VkImageMemoryBarrier2> ib;
  for (auto& o : outs)
    if (o.img && o.layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
      ib.push_back(image_barrier(o.img->img, o.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL));
  VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  mb.srcStageMask = ALL; mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  mb.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
  di.imageMemoryBarrierCount = (uint32_t)ib.size(); di.pImageMemoryBarriers = ib.data();
  vkCmdPipelineBarrier2(c.cb, &di);
  for (size_t i = 0; i < outs.size(); i++) {
    auto& o = outs[i];
    if (o.img) {
      VkBufferImageCopy r{};
      r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      r.imageExtent = {o.img->w, o.img->h, 1};
      vkCmdCopyImageToBuffer(c.cb, o.img->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, host[i].buf, 1, &r);
    } else {
      VkBufferCopy r{0, 0, o.buf->size};
      vkCmdCopyBuffer(c.cb, o.buf->buf, host[i].buf, 1, &r);
    }
  }
  // make the copies visible to the host (never elided: COVENANT pins HOST barriers)
  VkMemoryBarrier2 hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  hb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT; hb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  hb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT; hb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
  VkDependencyInfo hdi{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  hdi.memoryBarrierCount = 1; hdi.pMemoryBarriers = &hb;
  vkCmdPipelineBarrier2(c.cb, &hdi);
  submit_and_wait(c);
  for (size_t i = 0; i < outs.size(); i++) {
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)fnv1a(host[i].map, (size_t)host[i].size));
    c.hashes.push_back({outs[i].name, hex});
  }
}

inline double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline void report(Ctx& c) {
  std::string s = "{\"workload\": \"" + std::string(c.workload) + "\", \"device\": \"" + c.device_name + "\"";
  s += ", \"frames\": " + std::to_string(c.args.frames);
  char buf[64];
  snprintf(buf, sizeof buf, "%g", c.args.scale);
  s += ", \"scale\": " + std::string(buf);
  s += ", \"hashes\": {";
  for (size_t i = 0; i < c.hashes.size(); i++)
    s += (i ? ", " : "") + ("\"" + c.hashes[i].first + "\": \"" + c.hashes[i].second + "\"");
  s += "}";
  auto arr = [&](const char* k, const std::vector<double>& v) {
    s += std::string(", \"") + k + "\": [";
    for (size_t i = 0; i < v.size(); i++) { snprintf(buf, sizeof buf, "%s%.6f", i ? ", " : "", v[i]); s += buf; }
    s += "]";
  };
  arr("gpu_ms", c.gpu_ms);
  arr("cpu_ms", c.cpu_ms);
  snprintf(buf, sizeof buf, ", \"gpu_ms_median\": %.6f, \"cpu_ms_median\": %.6f", median(c.gpu_ms), median(c.cpu_ms));
  s += buf;
  s += ", \"validation_messages\": " + std::to_string(g_validation_errors);
  s += ", \"sync_hazards\": " + std::to_string(g_validation_sync_hazards);
  s += "}";
  if (!c.args.quiet) printf("%s\n", s.c_str());
  if (!c.args.json.empty()) {
    FILE* f = fopen(c.args.json.c_str(), "w");
    if (f) { fprintf(f, "%s\n", s.c_str()); fclose(f); }
  }
}

inline void finish(Ctx& c) {
  vkDeviceWaitIdle(c.dev);
  report(c);
  // Objects are not destroyed one by one: the process exits right after and
  // the driver reclaims everything (this also keeps validation output to the
  // messages that matter).
  fflush(stdout);
}

}  // namespace el
