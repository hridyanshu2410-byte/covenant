// Synchronization litmus programs (framebuffer-local dependencies and layout-transition
// order), run against the COVENANT model driver by tests/test_framebuffer_local.py.
//
//   sync_litmus <shader dir> <case>
//
// One render pass with two subpasses on a 64x64 color image A (and a second color
// image B): subpass 0 draws into A, subpass 1 draws into B and has A as an input
// attachment. The descriptor set has a storage buffer (binding 0) and the input
// attachment (binding 1). The case picks the shaders, the 0 -> 1 subpass dependency
// and its dependencyFlags. Shaders (compiled by the test): vs.spv, vs_write.spv
// (writes buf[0]), vs_read.spv, fs_write.spv (writes buf[0]), fs_read.spv, fs_input.spv
// (subpassLoad of A), fs_plain.spv. Only the "input" cases use A as an input attachment
// in subpass 1; everything else in each program is synchronized correctly.
//
// cases:
//   storage_byregion     fs_write -> [FRAGMENT/SHADER_WRITE -> FRAGMENT/SHADER_READ, BY_REGION] -> fs_read
//   storage_global       the same without BY_REGION
//   input_byregion       color write of A -> [COLOR_OUTPUT/COLOR_WRITE -> FRAGMENT/INPUT_READ, BY_REGION]
//                        -> subpassLoad(A)
//   vs_write_byregion    vs_write -> [VERTEX_SHADER/SHADER_WRITE -> FRAGMENT/SHADER_READ, BY_REGION] -> fs_read
//   vs_read_byregion     fs_write -> [FRAGMENT/SHADER_WRITE -> VERTEX_SHADER/SHADER_READ, BY_REGION] -> vs_read
//   storage_byregion_rp2 storage_byregion through vkCreateRenderPass2
//   mb2_global           vkCreateRenderPass2; the dependency's own masks are TOP -> BOTTOM (no access),
//                        a chained VkMemoryBarrier2 has FRAGMENT/SHADER_WRITE -> FRAGMENT/SHADER_READ
//   mv_storage_viewlocal multiview (views 0,1; A and B have 2 layers), VIEW_LOCAL, storage as above
//   mv_input_viewlocal   multiview, VIEW_LOCAL, input attachment as above
//   mv_input_viewoffset  multiview, VIEW_LOCAL with viewOffset 1, input attachment
//   barrier_byregion     two render passes (one subpass each, no dependencies) with a
//                        vkCmdPipelineBarrier FRAGMENT/SHADER_WRITE -> FRAGMENT/SHADER_READ, BY_REGION
//                        between them: fs_write in the first, fs_read in the second
//   barrier_global       the same without BY_REGION
//   rp_rp_transitions    two render passes on A, initial = final = SHADER_READ_ONLY, subpass layout
//                        COLOR_ATTACHMENT, no dependencies and nothing between them (a GTK shape)
//   rp_rp_ordered        rp_rp_transitions with EXTERNAL -> 0 and 0 -> EXTERNAL dependencies
//                        COLOR_OUTPUT/COLOR_WRITE -> COLOR_OUTPUT/COLOR_READ|WRITE in both passes
//   rp_barrier_transitions the first pass's final transition, then an image memory barrier
//                        SHADER_READ_ONLY -> COLOR_ATTACHMENT on A, then the second pass
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                        \
    VkResult r_ = (x);                                                        \
    if (r_ != VK_SUCCESS) {                                                   \
      fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, r_); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static std::vector<char> read_file(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { perror(path.c_str()); exit(1); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> data(n);
  if (fread(data.data(), 1, n, f) != (size_t)n) { perror(path.c_str()); exit(1); }
  fclose(f);
  return data;
}

static uint32_t find_memory(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  exit(1);
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: sync_litmus <shader dir> <case>\n"); return 2; }
  const std::string dir = argv[1], cs = argv[2];
  const bool multiview = cs.rfind("mv_", 0) == 0;
  const bool rp2 = multiview || cs == "storage_byregion_rp2" || cs == "mb2_global";
  const bool two_passes = cs.rfind("barrier_", 0) == 0 || cs.rfind("rp_", 0) == 0;
  const uint32_t layers = multiview ? 2 : 1;
  // subpass 1 has A as an input attachment only in the input cases (else A is not used there)
  const bool use_input = cs.find("input") != std::string::npos;

  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance instance;
  CHECK(vkCreateInstance(&ici, nullptr, &instance));
  uint32_t n = 1;
  VkPhysicalDevice pd;
  vkEnumeratePhysicalDevices(instance, &n, &pd);
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice device;
  CHECK(vkCreateDevice(pd, &dci, nullptr, &device));
  VkQueue queue;
  vkGetDeviceQueue(device, 0, 0, &queue);

  // ---- images A, B (color, 64x64, `layers` layers) and the storage buffer
  auto make_image = [&](VkImage* img, VkImageView* view) {
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {64, 64, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    CHECK(vkCreateImage(device, &ci, nullptr, img));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(device, *img, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_memory(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory mem;
    CHECK(vkAllocateMemory(device, &ai, nullptr, &mem));
    CHECK(vkBindImageMemory(device, *img, mem, 0));
    VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = *img;
    vci.viewType = multiview ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    CHECK(vkCreateImageView(device, &vci, nullptr, view));
  };
  VkImage imgA, imgB;
  VkImageView viewA, viewB;
  make_image(&imgA, &viewA);
  make_image(&imgB, &viewB);

  VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = 256;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  VkBuffer buf;
  CHECK(vkCreateBuffer(device, &bci, nullptr, &buf));
  VkMemoryRequirements bmr;
  vkGetBufferMemoryRequirements(device, buf, &bmr);
  VkMemoryAllocateInfo bai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  bai.allocationSize = bmr.size;
  bai.memoryTypeIndex = find_memory(pd, bmr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkDeviceMemory bmem;
  CHECK(vkAllocateMemory(device, &bai, nullptr, &bmem));
  CHECK(vkBindBufferMemory(device, buf, bmem, 0));

  // ---- descriptors: binding 0 storage buffer, binding 1 input attachment (A)
  VkDescriptorSetLayoutBinding binds[2] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
  VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 2;
  dslci.pBindings = binds;
  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl));
  VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1}};
  VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 2;
  dpci.pPoolSizes = ps;
  VkDescriptorPool dp;
  CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &dp));
  VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dp;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet set;
  CHECK(vkAllocateDescriptorSets(device, &dsai, &set));
  VkDescriptorBufferInfo dbi = {buf, 0, 256};
  VkDescriptorImageInfo dii = {VK_NULL_HANDLE, viewA, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
  w[0].dstSet = set; w[0].dstBinding = 0; w[0].descriptorCount = 1;
  w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &dbi;
  w[1].dstSet = set; w[1].dstBinding = 1; w[1].descriptorCount = 1;
  w[1].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT; w[1].pImageInfo = &dii;
  vkUpdateDescriptorSets(device, use_input ? 2 : 1, w, 0, nullptr);
  VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pl));

  // ---- render pass(es)
  const VkPipelineStageFlags FS = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VS = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                             CO = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubpassDependency dep = {0, 1, FS, FS, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                             VK_DEPENDENCY_BY_REGION_BIT};
  int32_t view_offset = 0;
  std::string vs0 = "vs", fs0 = "fs_write", vs1 = "vs", fs1 = "fs_read";
  if (cs == "storage_global") dep.dependencyFlags = 0;
  if (cs == "input_byregion" || cs == "mv_input_viewlocal" || cs == "mv_input_viewoffset") {
    dep = {0, 1, CO, FS, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
           VK_DEPENDENCY_BY_REGION_BIT};
    fs0 = "fs_plain";
    fs1 = "fs_input";
    // A's store operation (COLOR_ATTACHMENT_OUTPUT, in subpass 1) must see the transition too
    dep.dstStageMask |= CO;
    dep.dstAccessMask |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  }
  if (cs == "vs_write_byregion") { dep.srcStageMask = VS; vs0 = "vs_write"; fs0 = "fs_plain"; }
  if (cs == "vs_read_byregion") { dep.dstStageMask = VS; vs1 = "vs_read"; fs1 = "fs_plain"; }
  if (multiview) dep.dependencyFlags = VK_DEPENDENCY_VIEW_LOCAL_BIT;
  if (cs == "mv_input_viewoffset") view_offset = 1;
  if (cs == "barrier_byregion" || cs == "barrier_global") { fs0 = "fs_write"; fs1 = "fs_read"; }
  if (cs.rfind("rp_", 0) == 0) { fs0 = fs1 = "fs_plain"; }

  VkRenderPass rp, rp_b = VK_NULL_HANDLE;
  if (!two_passes) {
    VkAttachmentDescription att[2] = {};
    for (auto& a : att) {
      a.format = VK_FORMAT_R8G8B8A8_UNORM;
      a.samples = VK_SAMPLE_COUNT_1_BIT;
      a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
      a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      a.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      a.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    VkAttachmentReference a0c = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference a0i = {0, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkAttachmentReference a1c = {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp[2] = {};
    sp[0].pipelineBindPoint = sp[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp[0].colorAttachmentCount = 1; sp[0].pColorAttachments = &a0c;
    sp[1].colorAttachmentCount = 1; sp[1].pColorAttachments = &a1c;
    sp[1].inputAttachmentCount = use_input ? 1 : 0; sp[1].pInputAttachments = &a0i;
    if (!rp2) {
      VkRenderPassCreateInfo rci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
      rci.attachmentCount = 2; rci.pAttachments = att;
      rci.subpassCount = 2; rci.pSubpasses = sp;
      // dependency 1: carries B's UNDEFINED -> COLOR_ATTACHMENT transition into subpass 1
      VkSubpassDependency deps[2] = {dep, {0, 1, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, CO, 0,
                                           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0}};
      rci.dependencyCount = 2; rci.pDependencies = deps;
      CHECK(vkCreateRenderPass(device, &rci, nullptr, &rp));
    } else {
      VkAttachmentDescription2 att2[2] = {{VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2},
                                          {VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2}};
      for (int i = 0; i < 2; i++) {
        att2[i].format = att[i].format; att2[i].samples = att[i].samples;
        att2[i].loadOp = att[i].loadOp; att2[i].storeOp = att[i].storeOp;
        att2[i].stencilLoadOp = att[i].stencilLoadOp; att2[i].stencilStoreOp = att[i].stencilStoreOp;
        att2[i].initialLayout = att[i].initialLayout; att2[i].finalLayout = att[i].finalLayout;
      }
      VkAttachmentReference2 r0c = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 0,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT};
      VkAttachmentReference2 r0i = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 0,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT};
      VkAttachmentReference2 r1c = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 1,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT};
      VkSubpassDescription2 sp2[2] = {{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2},
                                      {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2}};
      sp2[0].pipelineBindPoint = sp2[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      sp2[0].colorAttachmentCount = 1; sp2[0].pColorAttachments = &r0c;
      sp2[1].colorAttachmentCount = 1; sp2[1].pColorAttachments = &r1c;
      sp2[1].inputAttachmentCount = use_input ? 1 : 0; sp2[1].pInputAttachments = &r0i;
      if (multiview) sp2[0].viewMask = sp2[1].viewMask = 3;
      VkSubpassDependency2 d2 = {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2};
      d2.srcSubpass = dep.srcSubpass; d2.dstSubpass = dep.dstSubpass;
      d2.srcStageMask = dep.srcStageMask; d2.dstStageMask = dep.dstStageMask;
      d2.srcAccessMask = dep.srcAccessMask; d2.dstAccessMask = dep.dstAccessMask;
      d2.dependencyFlags = dep.dependencyFlags;
      d2.viewOffset = view_offset;
      VkMemoryBarrier2 mb2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
      if (cs == "mb2_global") {
        // sync2: the chained barrier's masks replace the dependency's (which order nothing here)
        d2.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; d2.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        d2.srcAccessMask = d2.dstAccessMask = 0;
        d2.dependencyFlags = 0;
        mb2.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT; mb2.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        mb2.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT; mb2.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        d2.pNext = &mb2;
      }
      VkRenderPassCreateInfo2 rci2 = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2};
      rci2.attachmentCount = 2; rci2.pAttachments = att2;
      rci2.subpassCount = 2; rci2.pSubpasses = sp2;
      VkSubpassDependency2 d2s[2] = {d2, {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2}};
      d2s[1].srcSubpass = 0; d2s[1].dstSubpass = 1;
      d2s[1].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; d2s[1].dstStageMask = CO;
      d2s[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      rci2.dependencyCount = 2; rci2.pDependencies = d2s;
      CHECK(vkCreateRenderPass2(device, &rci2, nullptr, &rp));
    }
  } else {
    // two single-subpass render passes on A
    VkAttachmentDescription a = {};
    a.format = VK_FORMAT_R8G8B8A8_UNORM;
    a.samples = VK_SAMPLE_COUNT_1_BIT;
    a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    const bool rpx = cs.rfind("rp_", 0) == 0;
    a.initialLayout = rpx ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    a.finalLayout = rpx ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference ac = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp = {};
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1; sp.pColorAttachments = &ac;
    VkRenderPassCreateInfo rci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rci.attachmentCount = 1; rci.pAttachments = &a;
    rci.subpassCount = 1; rci.pSubpasses = &sp;
    // rp_rp_ordered: explicit EXTERNAL dependencies chain the passes (and their transitions)
    const VkAccessFlags CRW = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkSubpassDependency ext[2] = {{VK_SUBPASS_EXTERNAL, 0, CO, CO, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, CRW, 0},
                                  {0, VK_SUBPASS_EXTERNAL, CO, CO, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, CRW, 0}};
    if (cs == "rp_rp_ordered") { rci.dependencyCount = 2; rci.pDependencies = ext; }
    CHECK(vkCreateRenderPass(device, &rci, nullptr, &rp));
    if (cs == "rp_barrier_transitions") {
      a.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;    // the barrier did the transition
      CHECK(vkCreateRenderPass(device, &rci, nullptr, &rp_b));
    }
  }
  if (rp_b == VK_NULL_HANDLE) rp_b = rp;

  VkFramebufferCreateInfo fci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  VkImageView fbv[2] = {viewA, viewB};
  fci.renderPass = rp;
  fci.attachmentCount = two_passes ? 1 : 2;
  fci.pAttachments = fbv;
  fci.width = fci.height = 64;
  fci.layers = 1;
  VkFramebuffer fb;
  CHECK(vkCreateFramebuffer(device, &fci, nullptr, &fb));

  // ---- pipelines
  auto module = [&](const std::string& name) {
    std::vector<char> code = read_file(dir + "/" + name + ".spv");
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = (const uint32_t*)code.data();
    VkShaderModule m;
    CHECK(vkCreateShaderModule(device, &ci, nullptr, &m));
    return m;
  };
  auto pipeline = [&](VkRenderPass pass, uint32_t subpass, const std::string& vs, const std::string& fs) {
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = module(vs); st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = module(fs); st[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp = {0, 0, 64, 64, 0, 1};
    VkRect2D sc = {{0, 0}, {64, 64}};
    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp; vps.scissorCount = 1; vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState cba = {};
    cba.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    VkGraphicsPipelineCreateInfo gci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gci.stageCount = 2; gci.pStages = st;
    gci.pVertexInputState = &vi; gci.pInputAssemblyState = &ia; gci.pViewportState = &vps;
    gci.pRasterizationState = &rs; gci.pMultisampleState = &ms; gci.pColorBlendState = &cb;
    gci.layout = pl; gci.renderPass = pass; gci.subpass = subpass;
    VkPipeline p;
    CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gci, nullptr, &p));
    return p;
  };
  VkPipeline p0 = pipeline(rp, 0, vs0, fs0);
  VkPipeline p1 = two_passes ? pipeline(rp_b, 0, vs1, fs1) : pipeline(rp, 1, vs1, fs1);

  // ---- record and submit
  VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  VkCommandPool pool;
  CHECK(vkCreateCommandPool(device, &cpci, nullptr, &pool));
  VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));
  VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  CHECK(vkBeginCommandBuffer(cmd, &bi));
  if (cs.rfind("rp_", 0) == 0) {
    // bring A to SHADER_READ_ONLY first (an image memory barrier transition)
    VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask = 0; ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = imgA; ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, CO, 0, 0, nullptr, 0, nullptr, 1, &ib);
  } else if (two_passes) {
    VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; ib.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = imgA; ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, CO, 0, 0, nullptr, 0, nullptr, 1, &ib);
  }
  VkClearValue clears[2] = {};
  VkRenderPassBeginInfo rbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rbi.renderPass = rp;
  rbi.framebuffer = fb;
  rbi.renderArea = {{0, 0}, {64, 64}};
  rbi.clearValueCount = two_passes ? 1 : 2;
  rbi.pClearValues = clears;
  vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p0);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  if (!two_passes) {
    vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
  } else {
    vkCmdEndRenderPass(cmd);
    if (cs == "barrier_byregion" || cs == "barrier_global") {
      VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, FS | CO, FS | CO, cs == "barrier_byregion" ? VK_DEPENDENCY_BY_REGION_BIT : 0, 1, &mb,
                           0, nullptr, 0, nullptr);
    }
    if (cs == "rp_barrier_transitions") {
      VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      ib.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      ib.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
      ib.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; ib.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      ib.image = imgA; ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdPipelineBarrier(cmd, CO, CO, 0, 0, nullptr, 0, nullptr, 1, &ib);
    }
    rbi.renderPass = rp_b;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 0, nullptr);
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p1);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRenderPass(cmd);
  CHECK(vkEndCommandBuffer(cmd));
  VkFenceCreateInfo fnci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  CHECK(vkCreateFence(device, &fnci, nullptr, &fence));
  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  CHECK(vkQueueSubmit(queue, 1, &si, fence));
  CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
  vkDeviceWaitIdle(device);
  return 0;
}
