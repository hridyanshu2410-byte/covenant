// A minimal but realistic headless Vulkan renderer (plain C API).
//
//   triangle <tri.vert.spv> <tri.frag.spv> [--frames N] [--bug-no-fence-wait] [--events] [--bug-event-early]
//
// Headless surface + swapchain, one graphics pipeline with dynamic rendering,
// 2 frames in flight, a persistently mapped uniform buffer per frame in flight
// written every frame, per-frame fences and acquire semaphores, per-image
// render-finished semaphores. --bug-no-fence-wait skips the per-frame fence
// wait, so the CPU may overwrite a uniform buffer the GPU is still reading.
#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                        \
    VkResult r_ = (x);                                                        \
    if (r_ != VK_SUCCESS) {                                                   \
      fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, r_); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static const int kFramesInFlight = 2;

struct FrameUniforms { float transform[16]; float tint[4]; };

static std::vector<char> read_file(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> data(n);
  if (fread(data.data(), 1, n, f) != (size_t)n) { perror(path); exit(1); }
  fclose(f);
  return data;
}

static uint32_t find_memory(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  fprintf(stderr, "no suitable memory type\n");
  exit(1);
}

int main(int argc, char** argv) {
  const char* vert_path = nullptr;
  const char* frag_path = nullptr;
  bool bug_no_fence_wait = false;
  bool use_events = false, event_early = false;
  // --clear-only replaces the draw by vkCmdClearAttachments (load op LOAD); --query-copy
  // copies a timestamp into a per-slot results buffer; --bug-query-copy-vertices copies it into
  // the vertex buffer instead, which the next frame's draw reads with no barrier in between
  bool clear_only = false, query_copy = false, query_into_vertices = false;
  int frames = 6;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--bug-no-fence-wait")) bug_no_fence_wait = true;
    // --events: the present transition is a split barrier (vkCmdSetEvent2 after the
    // pass, vkCmdWaitEvents2 before submit); --bug-event-early sets the event before
    // the pass, so the color writes are outside the wait's first scope
    else if (!strcmp(argv[i], "--events")) use_events = true;
    else if (!strcmp(argv[i], "--bug-event-early")) use_events = event_early = true;
    else if (!strcmp(argv[i], "--clear-only")) clear_only = true;
    else if (!strcmp(argv[i], "--query-copy")) query_copy = true;
    else if (!strcmp(argv[i], "--bug-query-copy-vertices")) query_copy = query_into_vertices = true;
    else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
    else if (!vert_path) vert_path = argv[i];
    else frag_path = argv[i];
  }
  if (!vert_path || !frag_path) {
    fprintf(stderr, "usage: %s tri.vert.spv tri.frag.spv [--frames N] [--bug-no-fence-wait]\n", argv[0]);
    return 2;
  }

  // ---- instance, headless surface
  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "covenant-triangle";
  app.apiVersion = VK_API_VERSION_1_3;
  const char* inst_ext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
                            VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = 3;
  ici.ppEnabledExtensionNames = inst_ext;
  VkInstance instance;
  CHECK(vkCreateInstance(&ici, nullptr, &instance));

  auto pfnCreateHeadless = (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT");
  auto pfnSetName = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT");
  VkHeadlessSurfaceCreateInfoEXT hsci = {VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT};
  VkSurfaceKHR surface;
  CHECK(pfnCreateHeadless(instance, &hsci, nullptr, &surface));

  // ---- physical device, queue family
  uint32_t n = 0;
  CHECK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
  std::vector<VkPhysicalDevice> pds(n);
  CHECK(vkEnumeratePhysicalDevices(instance, &n, pds.data()));
  VkPhysicalDevice pd = pds[0];
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
  std::vector<VkQueueFamilyProperties> qfp(n);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, qfp.data());
  uint32_t family = UINT32_MAX;
  for (uint32_t i = 0; i < n; i++) {
    VkBool32 present = VK_FALSE;
    CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present));
    if ((qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) { family = i; break; }
  }
  if (family == UINT32_MAX) { fprintf(stderr, "no graphics+present queue\n"); return 1; }

  // ---- device with dynamic rendering + synchronization2
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.dynamicRendering = VK_TRUE;
  f13.synchronization2 = VK_TRUE;
  const char* dev_ext[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &f13;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 1;
  dci.ppEnabledExtensionNames = dev_ext;
  VkDevice device;
  CHECK(vkCreateDevice(pd, &dci, nullptr, &device));
  VkQueue queue;
  vkGetDeviceQueue(device, family, 0, &queue);

  auto set_name = [&](VkObjectType type, uint64_t handle, const char* name) {
    VkDebugUtilsObjectNameInfoEXT ni = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    ni.objectType = type;
    ni.objectHandle = handle;
    ni.pObjectName = name;
    if (pfnSetName) pfnSetName(device, &ni);
  };

  // ---- swapchain
  VkSurfaceCapabilitiesKHR caps;
  CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps));
  VkExtent2D extent = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : VkExtent2D{640, 480};
  CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &n, nullptr));
  std::vector<VkSurfaceFormatKHR> formats(n);
  CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &n, formats.data()));
  VkSurfaceFormatKHR sfmt = formats[0];
  for (auto& f : formats)
    if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) sfmt = f;
  VkSwapchainCreateInfoKHR sci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sci.surface = surface;
  sci.minImageCount = caps.minImageCount + 1;
  if (caps.maxImageCount && sci.minImageCount > caps.maxImageCount) sci.minImageCount = caps.maxImageCount;
  sci.imageFormat = sfmt.format;
  sci.imageColorSpace = sfmt.colorSpace;
  sci.imageExtent = extent;
  sci.imageArrayLayers = 1;
  sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  sci.preTransform = caps.currentTransform;
  sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sci.clipped = VK_TRUE;
  VkSwapchainKHR swapchain;
  CHECK(vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain));
  CHECK(vkGetSwapchainImagesKHR(device, swapchain, &n, nullptr));
  std::vector<VkImage> sc_images(n);
  CHECK(vkGetSwapchainImagesKHR(device, swapchain, &n, sc_images.data()));
  std::vector<VkImageView> sc_views(n);
  for (uint32_t i = 0; i < n; i++) {
    VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = sc_images[i];
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = sfmt.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    CHECK(vkCreateImageView(device, &vci, nullptr, &sc_views[i]));
  }

  // ---- descriptor set layout, pipeline layout, pipeline
  VkDescriptorSetLayoutBinding ubo_binding = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 1;
  dslci.pBindings = &ubo_binding;
  VkDescriptorSetLayout set_layout;
  CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &set_layout));
  VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &set_layout;
  VkPipelineLayout pipeline_layout;
  CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));

  VkShaderModule modules[2];
  const char* paths[2] = {vert_path, frag_path};
  for (int i = 0; i < 2; i++) {
    std::vector<char> code = read_file(paths[i]);
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode = (const uint32_t*)code.data();
    CHECK(vkCreateShaderModule(device, &smci, nullptr, &modules[i]));
  }
  VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                               {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = modules[0];
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = modules[1];
  stages[1].pName = "main";

  VkVertexInputBindingDescription vb = {0, 5 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription va[2] = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
                                             {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 2 * sizeof(float)}};
  VkPipelineVertexInputStateCreateInfo vis = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vis.vertexBindingDescriptionCount = 1;
  vis.pVertexBindingDescriptions = &vb;
  vis.vertexAttributeDescriptionCount = 2;
  vis.pVertexAttributeDescriptions = va;
  VkPipelineInputAssemblyStateCreateInfo ias = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ias.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vps.viewportCount = 1;
  vps.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState cba = {};
  cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1;
  cbs.pAttachments = &cba;
  VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  ds.dynamicStateCount = 2;
  ds.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo prci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  prci.colorAttachmentCount = 1;
  prci.pColorAttachmentFormats = &sfmt.format;
  VkGraphicsPipelineCreateInfo gpci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gpci.pNext = &prci;
  gpci.stageCount = 2;
  gpci.pStages = stages;
  gpci.pVertexInputState = &vis;
  gpci.pInputAssemblyState = &ias;
  gpci.pViewportState = &vps;
  gpci.pRasterizationState = &rs;
  gpci.pMultisampleState = &ms;
  gpci.pColorBlendState = &cbs;
  gpci.pDynamicState = &ds;
  gpci.layout = pipeline_layout;
  VkPipeline pipeline;
  CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline));

  // ---- vertex buffer (host-visible, written once)
  const float vertices[] = {0.0f, -0.5f, 1, 0, 0, 0.5f, 0.5f, 0, 1, 0, -0.5f, 0.5f, 0, 0, 1};
  auto make_buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer* buf, VkDeviceMemory* mem) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(vkCreateBuffer(device, &bci, nullptr, buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, *buf, &req);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory(pd, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK(vkAllocateMemory(device, &mai, nullptr, mem));
    CHECK(vkBindBufferMemory(device, *buf, *mem, 0));
  };
  VkBuffer vbuf;
  VkDeviceMemory vmem;
  make_buffer(sizeof vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vbuf, &vmem);
  set_name(VK_OBJECT_TYPE_BUFFER, (uint64_t)vbuf, "vertices");
  void* vptr;
  CHECK(vkMapMemory(device, vmem, 0, VK_WHOLE_SIZE, 0, &vptr));
  memcpy(vptr, vertices, sizeof vertices);
  vkUnmapMemory(device, vmem);

  // ---- per frame in flight: uniform buffer (persistently mapped), descriptor set, cmd buffer, sync
  VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight};
  VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = kFramesInFlight;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_size;
  VkDescriptorPool dpool;
  CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &dpool));
  VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpci.queueFamilyIndex = family;
  VkCommandPool cpool;
  CHECK(vkCreateCommandPool(device, &cpci, nullptr, &cpool));

  VkBuffer ubo[kFramesInFlight];
  VkDeviceMemory ubo_mem[kFramesInFlight];
  FrameUniforms* ubo_ptr[kFramesInFlight];
  VkDescriptorSet sets[kFramesInFlight];
  VkCommandBuffer cbs_[kFramesInFlight];
  VkFence in_flight[kFramesInFlight];
  VkEvent events[kFramesInFlight];
  VkSemaphore image_available[kFramesInFlight];
  VkDescriptorSetLayout layouts[kFramesInFlight] = {set_layout, set_layout};
  VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dpool;
  dsai.descriptorSetCount = kFramesInFlight;
  dsai.pSetLayouts = layouts;
  CHECK(vkAllocateDescriptorSets(device, &dsai, sets));
  VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = cpool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = kFramesInFlight;
  CHECK(vkAllocateCommandBuffers(device, &cbai, cbs_));
  for (int k = 0; k < kFramesInFlight; k++) {
    make_buffer(sizeof(FrameUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &ubo[k], &ubo_mem[k]);
    char name[16];
    snprintf(name, sizeof name, "ubo[%d]", k);
    set_name(VK_OBJECT_TYPE_BUFFER, (uint64_t)ubo[k], name);
    CHECK(vkMapMemory(device, ubo_mem[k], 0, VK_WHOLE_SIZE, 0, (void**)&ubo_ptr[k]));
    VkDescriptorBufferInfo dbi = {ubo[k], 0, sizeof(FrameUniforms)};
    VkWriteDescriptorSet wds = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet = sets[k];
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    CHECK(vkCreateFence(device, &fci, nullptr, &in_flight[k]));
    VkEventCreateInfo eci = {VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
    CHECK(vkCreateEvent(device, &eci, nullptr, &events[k]));
    VkSemaphoreCreateInfo semci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    CHECK(vkCreateSemaphore(device, &semci, nullptr, &image_available[k]));
  }
  VkQueryPool qpool = VK_NULL_HANDLE;
  VkBuffer qres[kFramesInFlight];
  VkDeviceMemory qres_mem[kFramesInFlight];
  if (query_copy) {
    VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = kFramesInFlight;
    CHECK(vkCreateQueryPool(device, &qpci, nullptr, &qpool));
    for (int k = 0; k < kFramesInFlight; k++) {
      make_buffer(8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &qres[k], &qres_mem[k]);
      char name[16];
      snprintf(name, sizeof name, "qres[%d]", k);
      set_name(VK_OBJECT_TYPE_BUFFER, (uint64_t)qres[k], name);
    }
  }
  std::vector<VkSemaphore> render_done(sc_images.size());
  for (auto& s : render_done) {
    VkSemaphoreCreateInfo semci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    CHECK(vkCreateSemaphore(device, &semci, nullptr, &s));
  }

  // ---- frame loop
  for (int f = 0; f < frames; f++) {
    int k = f % kFramesInFlight;
    if (!bug_no_fence_wait) CHECK(vkWaitForFences(device, 1, &in_flight[k], VK_TRUE, UINT64_MAX));
    uint32_t idx;
    CHECK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, image_available[k], VK_NULL_HANDLE, &idx));
    CHECK(vkResetFences(device, 1, &in_flight[k]));

    FrameUniforms u = {};
    float a = 0.3f * f;
    u.transform[0] = cosf(a); u.transform[1] = sinf(a);
    u.transform[4] = -sinf(a); u.transform[5] = cosf(a);
    u.transform[10] = 1; u.transform[15] = 1;
    u.tint[0] = u.tint[1] = u.tint[2] = u.tint[3] = 1;
    memcpy(ubo_ptr[k], &u, sizeof u);          // host write to the persistently mapped UBO

    VkCommandBuffer cb = cbs_[k];
    CHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo cbbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cb, &cbbi));

    VkImageMemoryBarrier2 to_color = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    to_color.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_color.srcAccessMask = 0;
    to_color.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_color.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                             (clear_only ? VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT : 0);   // load op LOAD
    to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_color.srcQueueFamilyIndex = to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_color.image = sc_images[idx];
    to_color.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dep = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &to_color;
    vkCmdPipelineBarrier2(cb, &dep);

    VkRenderingAttachmentInfo color = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = sc_views[idx];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = clear_only ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.1f, 0.1f, 0.1f, 1.0f}};
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    VkImageMemoryBarrier2 to_present = to_color;
    to_present.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_present.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    to_present.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    to_present.dstAccessMask = 0;
    to_present.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkDependencyInfo pdep = dep;
    pdep.pImageMemoryBarriers = &to_present;
    if (use_events) vkCmdResetEvent2(cb, events[k], VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    if (event_early) vkCmdSetEvent2(cb, events[k], &pdep);
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkViewport vp = {0, 0, (float)extent.width, (float)extent.height, 0, 1};
    VkRect2D sc = {{0, 0}, extent};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &sets[k], 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &vbuf, &off);
    if (clear_only) {
      VkClearAttachment ca = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {}};
      ca.clearValue.color = {{0.1f, 0.1f, 0.1f, 1.0f}};
      VkClearRect cr = {{{0, 0}, extent}, 0, 1};
      vkCmdClearAttachments(cb, 1, &ca, 1, &cr);
    } else {
      vkCmdDraw(cb, 3, 1, 0, 0);
    }
    vkCmdEndRendering(cb);
    if (query_copy) {
      vkCmdResetQueryPool(cb, qpool, k, 1);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, k);
      vkCmdCopyQueryPoolResults(cb, qpool, k, 1, query_into_vertices ? vbuf : qres[k], 0, 8,
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    }

    if (use_events) {
      if (!event_early) vkCmdSetEvent2(cb, events[k], &pdep);
      vkCmdWaitEvents2(cb, 1, &events[k], &pdep);
    } else {
      vkCmdPipelineBarrier2(cb, &pdep);
    }
    CHECK(vkEndCommandBuffer(cb));

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &image_available[k];
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &render_done[idx];
    CHECK(vkQueueSubmit(queue, 1, &si, in_flight[k]));

    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &render_done[idx];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain;
    pi.pImageIndices = &idx;
    CHECK(vkQueuePresentKHR(queue, &pi));
  }
  CHECK(vkDeviceWaitIdle(device));

  // ---- teardown
  for (int k = 0; k < kFramesInFlight; k++) {
    vkDestroyFence(device, in_flight[k], nullptr);
    vkDestroyEvent(device, events[k], nullptr);
    vkDestroySemaphore(device, image_available[k], nullptr);
    vkUnmapMemory(device, ubo_mem[k]);
    vkDestroyBuffer(device, ubo[k], nullptr);
    vkFreeMemory(device, ubo_mem[k], nullptr);
  }
  for (auto s : render_done) vkDestroySemaphore(device, s, nullptr);
  vkDestroyCommandPool(device, cpool, nullptr);
  vkDestroyDescriptorPool(device, dpool, nullptr);
  vkDestroyBuffer(device, vbuf, nullptr);
  vkFreeMemory(device, vmem, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  for (auto m : modules) vkDestroyShaderModule(device, m, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
  for (auto v : sc_views) vkDestroyImageView(device, v, nullptr);
  vkDestroySwapchainKHR(device, swapchain, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroySurfaceKHR(instance, surface, nullptr);
  vkDestroyInstance(instance, nullptr);
  printf("rendered %d frames%s\n", frames, bug_no_fence_wait ? " (bug: no fence wait)" : "");
  return 0;
}
