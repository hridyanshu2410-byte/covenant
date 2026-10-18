// wf.cpp - whole-frame Vulkan workloads for the COVENANT vs syncval differential test.
//
// One program, three workloads, each a small but complete frame loop with a
// swapchain (VK_EXT_headless_surface), frames in flight, per-frame and shared
// resources, host writes to mapped memory, fences, binary and timeline
// semaphores, and (W2 --mq) two queue families with queue-family ownership
// transfers. The same binary runs on lavapipe (+ Khronos validation) and on the
// COVENANT model driver (libcovvk), so both oracles see the same API stream.
//
//   wf offscreen [--fif 2|3]   W1: compute -> shared SSBO, draw to per-frame offscreen
//                                  image, blit to the swapchain image; fences; per-frame UBO
//   wf timeline [--mq]         W2: compute submit (signals timeline tlc) -> graphics submit
//                                  (waits tlc, signals timeline tlg); host paces on tlg
//                                  with vkWaitSemaphores; --mq puts compute on another
//                                  queue family (EXCLUSIVE buffers + QFOT)
//   wf upload                  W3: host fills a per-frame region of a staging ring, copy to a
//                                  shared vertex buffer, depth-tested draw straight into
//                                  the swapchain image, shared depth image; fences
//   wf <w> --sites             print the mutation catalogue (JSON) and exit
//
// Mutations: WF_MUT="site:kind@frame;site:kind@*" (frame = 0-based frame number or *).
// Every applied mutation is printed to stderr as "MUT-APPLIED site:kind@frame".
// The acquired image index of every frame is printed as "ACQ frame image".
#define VK_USE_PLATFORM_XCB_KHR
#include <xcb/xcb.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#define CK(x)                                                                         \
  do {                                                                                \
    VkResult r_ = (x);                                                                \
    if (r_ < 0) { fprintf(stderr, "FATAL %s = %d at line %d\n", #x, r_, __LINE__); exit(2); } \
  } while (0)

// ---------------------------------------------------------------- mutations
struct Mut { std::string site, kind; int frame; };
static std::vector<Mut> g_muts;
static int g_frame = -1;
static std::set<std::string> g_applied;

static void parse_muts() {
  const char* e = getenv("WF_MUT");
  if (!e) return;
  std::string s(e);
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(';', p);
    if (q == std::string::npos) q = s.size();
    std::string t = s.substr(p, q - p);
    p = q + 1;
    if (t.empty()) continue;
    size_t c = t.find(':'), a = t.find('@');
    if (c == std::string::npos) continue;
    Mut m;
    m.site = t.substr(0, c);
    m.kind = t.substr(c + 1, (a == std::string::npos ? t.size() : a) - c - 1);
    m.frame = (a == std::string::npos || t.substr(a + 1) == "*") ? -1 : atoi(t.c_str() + a + 1);
    g_muts.push_back(m);
  }
}
// kind of the mutation of `site` that applies in the current frame, or "" (first match wins)
static std::string mut(const std::string& site) {
  for (auto& m : g_muts)
    if (m.site == site && (m.frame < 0 || m.frame == g_frame)) {
      char b[200];
      snprintf(b, sizeof b, "%s:%s@%d", m.site.c_str(), m.kind.c_str(), g_frame);
      if (g_applied.insert(b).second) fprintf(stderr, "MUT-APPLIED %s\n", b);
      return m.kind;
    }
  return "";
}

// ---------------------------------------------------------------- globals
static VkInstance inst;
static VkPhysicalDevice pd;
static VkDevice dev;
static uint32_t gfam = 0, cfam = 0;
static VkQueue gq, cq;
static bool mq = false;
static VkSurfaceKHR surf;
static VkSwapchainKHR sc;
static VkFormat sc_fmt;
static VkExtent2D sc_ext;
static std::vector<VkImage> sc_img;
static std::vector<VkImageView> sc_view;
static PFN_vkSetDebugUtilsObjectNameEXT pSetName;
static VkPhysicalDeviceMemoryProperties memprops;
static std::string g_shaders = ".";

static void name(VkObjectType t, uint64_t h, const std::string& n) {
  if (!pSetName) return;
  VkDebugUtilsObjectNameInfoEXT ni{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
  ni.objectType = t; ni.objectHandle = h; ni.pObjectName = n.c_str();
  pSetName(dev, &ni);
}
#define NAME(t, h, n) name(t, (uint64_t)(h), n)

static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want) return i;
  fprintf(stderr, "FATAL no memory type\n"); exit(2);
}

struct Buf { VkBuffer b; VkDeviceMemory m; void* map = nullptr; VkDeviceSize size; };
static Buf mkbuf(const std::string& n, VkDeviceSize size, VkBufferUsageFlags usage, bool host, bool exclusive_mq = false) {
  Buf r; r.size = size;
  VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ci.size = size; ci.usage = usage; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  uint32_t fams[2] = {gfam, cfam};
  if (mq && !exclusive_mq && gfam != cfam) { ci.sharingMode = VK_SHARING_MODE_CONCURRENT; ci.queueFamilyIndexCount = 2; ci.pQueueFamilyIndices = fams; }
  CK(vkCreateBuffer(dev, &ci, nullptr, &r.b));
  VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, r.b, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = memtype(req.memoryTypeBits, host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                                        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  CK(vkAllocateMemory(dev, &ai, nullptr, &r.m));
  CK(vkBindBufferMemory(dev, r.b, r.m, 0));
  if (host) CK(vkMapMemory(dev, r.m, 0, VK_WHOLE_SIZE, 0, &r.map));
  NAME(VK_OBJECT_TYPE_BUFFER, r.b, n);
  NAME(VK_OBJECT_TYPE_DEVICE_MEMORY, r.m, n + ".mem");
  return r;
}
struct Img { VkImage i; VkDeviceMemory m; VkImageView v; VkFormat f; VkExtent2D e; };
static Img mkimg(const std::string& n, VkFormat f, VkExtent2D e, VkImageUsageFlags usage, VkImageAspectFlags asp) {
  Img r; r.f = f; r.e = e;
  VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ci.imageType = VK_IMAGE_TYPE_2D; ci.format = f; ci.extent = {e.width, e.height, 1};
  ci.mipLevels = 1; ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = usage; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE; ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  CK(vkCreateImage(dev, &ci, nullptr, &r.i));
  VkMemoryRequirements req; vkGetImageMemoryRequirements(dev, r.i, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size; ai.memoryTypeIndex = memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  CK(vkAllocateMemory(dev, &ai, nullptr, &r.m));
  CK(vkBindImageMemory(dev, r.i, r.m, 0));
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = r.i; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = f; vi.subresourceRange = {asp, 0, 1, 0, 1};
  CK(vkCreateImageView(dev, &vi, nullptr, &r.v));
  NAME(VK_OBJECT_TYPE_IMAGE, r.i, n);
  NAME(VK_OBJECT_TYPE_IMAGE_VIEW, r.v, n + ".view");
  return r;
}

static std::vector<char> readfile(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "FATAL cannot open %s\n", p.c_str()); exit(2); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> v(n); if (fread(v.data(), 1, n, f) != (size_t)n) exit(2); fclose(f); return v;
}
static VkShaderModule shader(const char* file) {
  auto code = readfile(g_shaders + "/" + file);
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = code.size(); ci.pCode = (const uint32_t*)code.data();
  VkShaderModule m; CK(vkCreateShaderModule(dev, &ci, nullptr, &m)); return m;
}

// ---------------------------------------------------------------- barriers with mutation sites
// A barrier site: correct masks plus a plausible-but-wrong alternative per side.
struct Site {
  const char* site;
  VkPipelineStageFlags2 ss; VkAccessFlags2 sa; VkPipelineStageFlags2 ds; VkAccessFlags2 da;
  VkPipelineStageFlags2 ss_w; VkAccessFlags2 sa_w; VkPipelineStageFlags2 ds_w; VkAccessFlags2 da_w;
  bool transition;   // layout transition or QFOT half: never dropped by "drop"
};
static void apply_site(const Site& s, VkPipelineStageFlags2& ss, VkAccessFlags2& sa, VkPipelineStageFlags2& ds,
                       VkAccessFlags2& da, bool& drop) {
  ss = s.ss; sa = s.sa; ds = s.ds; da = s.da; drop = false;
  // manual experiment only (not a catalogue mutation): force the present transition's dst stage
  if (s.site == std::string("b_sc_present"))
    if (const char* e = getenv("WF_PRESENT_DST_HEX")) ds = strtoull(e, nullptr, 16);
  std::string k = mut(s.site);
  if (k.empty()) return;
  if (k == "drop") drop = true;
  else if (k == "src_none") { ss = VK_PIPELINE_STAGE_2_NONE; sa = VK_ACCESS_2_NONE; }
  else if (k == "src_wrong") { ss = s.ss_w; sa = s.sa_w; }
  else if (k == "src_access_none") sa = VK_ACCESS_2_NONE;
  else if (k == "dst_none") { ds = VK_PIPELINE_STAGE_2_NONE; da = VK_ACCESS_2_NONE; }
  else if (k == "dst_wrong") { ds = s.ds_w; da = s.da_w; }
  else if (k == "dst_access_none") da = VK_ACCESS_2_NONE;
}
static void bar_img(VkCommandBuffer cb, const Site& s, VkImage img, VkImageLayout o, VkImageLayout n,
                    VkImageAspectFlags asp = VK_IMAGE_ASPECT_COLOR_BIT) {
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  bool drop;
  apply_site(s, b.srcStageMask, b.srcAccessMask, b.dstStageMask, b.dstAccessMask, drop);
  if (drop) return;
  b.oldLayout = o; b.newLayout = n; b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img; b.subresourceRange = {asp, 0, 1, 0, 1};
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
  vkCmdPipelineBarrier2(cb, &di);
}
static void bar_buf(VkCommandBuffer cb, const Site& s, VkBuffer buf, uint32_t sf = VK_QUEUE_FAMILY_IGNORED,
                    uint32_t df = VK_QUEUE_FAMILY_IGNORED) {
  VkBufferMemoryBarrier2 b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
  bool drop;
  apply_site(s, b.srcStageMask, b.srcAccessMask, b.dstStageMask, b.dstAccessMask, drop);
  if (drop) return;
  b.srcQueueFamilyIndex = sf; b.dstQueueFamilyIndex = df; b.buffer = buf; b.offset = 0; b.size = VK_WHOLE_SIZE;
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.bufferMemoryBarrierCount = 1; di.pBufferMemoryBarriers = &b;
  vkCmdPipelineBarrier2(cb, &di);
}

#define S2(x) VK_PIPELINE_STAGE_2_##x##_BIT
#define A2(x) VK_ACCESS_2_##x##_BIT
static const VkPipelineStageFlags2 NONE_S = VK_PIPELINE_STAGE_2_NONE;
static const VkAccessFlags2 NONE_A = VK_ACCESS_2_NONE;

// ---------------------------------------------------------------- mutation catalogue
struct CatEntry { const char* site; std::vector<const char*> kinds; const char* what; };
static const std::vector<const char*> BAR_K = {"drop", "src_none", "src_wrong", "src_access_none", "dst_none", "dst_wrong", "dst_access_none"};
static const std::vector<const char*> LT_K = {"src_none", "src_wrong", "src_access_none", "dst_none", "dst_wrong", "dst_access_none"};

static std::vector<CatEntry> catalogue(const std::string& w) {
  std::vector<CatEntry> c;
  // frame-level synchronization shared by all workloads
  if (w == "offscreen") c.push_back({"acq_wait", {"drop", "stage_none", "stage_wrong"}, "submit's wait on the acquire semaphore"});
  else c.push_back({"acq_wait", {"drop", "stage_none", "stage_wrong", "stage_early"}, "submit's wait on the acquire semaphore"});
  c.push_back({"acq_sem", {"shared"}, "one acquire semaphore for every frame slot"});
  c.push_back({"present_wait", {"drop"}, "present waits on no semaphore"});
  c.push_back({"render_sem", {"per_frame", "shared"}, "render-finished semaphore indexed by frame slot / one for all"});
  c.push_back({"signal_stage", {"none", "wrong", "late"}, "stage mask of the render-finished semaphore signal"});
  c.push_back({"cb", {"shared"}, "one command buffer re-recorded every frame"});
  if (w == "offscreen" || w == "upload") c.push_back({"fence_wait", {"drop", "other_slot"}, "host wait on the frame slot's fence"});
  if (w == "offscreen") {
    c.push_back({"ubo", {"shared"}, "per-frame uniform buffer indexed by 0"});
    c.push_back({"off", {"shared"}, "per-frame offscreen image indexed by 0"});
    c.push_back({"present", {"skip"}, "image acquired and rendered but not presented"});
    for (const char* s : {"b_sbuf_war", "b_sbuf_raw"}) c.push_back({s, BAR_K, "buffer barrier"});
    for (const char* s : {"b_off_color", "b_off_src", "b_sc_dst", "b_sc_present"}) c.push_back({s, LT_K, "image layout transition"});
  }
  if (w == "timeline") {
    c.push_back({"params", {"shared"}, "per-frame host-written params buffer indexed by 0"});
    c.push_back({"sim", {"shared"}, "per-frame simulation buffer indexed by 0"});
    c.push_back({"tl_gfx_wait", {"drop", "value_prev", "stage_wrong", "stage_early", "stage_all"}, "graphics submit's wait on timeline tlc"});
    c.push_back({"tl_host_wait", {"drop", "value_prev"}, "host vkWaitSemaphores on timeline tlg (frame pacing)"});
    c.push_back({"tl_comp_wait", {"drop", "value_prev", "stage_wrong"}, "compute submit's wait on timeline tlg (WAR protection)"});
    c.push_back({"tl_comp_signal", {"value_next"}, "compute submit signals tlc one value too high"});
    // on a compute-only family the graphics-stage alternatives would be invalid usage (VUID-...-09675)
    if (mq) c.push_back({"b_state", {"drop", "src_none", "src_access_none", "dst_none", "dst_access_none"}, "buffer barrier on the shared state buffer"});
    else c.push_back({"b_state", BAR_K, "buffer barrier on the shared state buffer"});
    for (const char* s : {"b_sc_color", "b_sc_present"}) c.push_back({s, LT_K, "image layout transition"});
    if (mq) {
      c.push_back({"qfot_rel", {"drop", "src_none", "src_access_none"}, "release half of the sim QFOT (compute queue)"});
      c.push_back({"qfot_acq", {"drop", "dst_none", "dst_wrong", "dst_access_none"}, "acquire half of the sim QFOT (graphics queue)"});
    }
  }
  if (w == "upload") {
    c.push_back({"ring", {"shared"}, "per-frame region of the staging ring indexed by 0"});
    for (const char* s : {"b_vbuf_war", "b_vbuf_raw"}) c.push_back({s, BAR_K, "buffer barrier"});
    for (const char* s : {"b_depth", "b_sc_color", "b_sc_present"}) c.push_back({s, LT_K, "image layout transition"});
  }
  return c;
}

// ---------------------------------------------------------------- device setup
static void init_device() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "wf"; app.apiVersion = VK_API_VERSION_1_3;
  bool xcb = getenv("WF_XCB") && atoi(getenv("WF_XCB")) != 0;
  const char* iext[] = {"VK_KHR_surface", xcb ? "VK_KHR_xcb_surface" : "VK_EXT_headless_surface", "VK_EXT_debug_utils"};
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app; ici.enabledExtensionCount = 3; ici.ppEnabledExtensionNames = iext;
  CK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &pd);
  vkGetPhysicalDeviceMemoryProperties(pd, &memprops);
  uint32_t nf = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nf); vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, qf.data());
  gfam = UINT32_MAX; cfam = UINT32_MAX;
  for (uint32_t i = 0; i < nf; i++) if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && gfam == UINT32_MAX) gfam = i;
  if (mq) {
    for (uint32_t i = 0; i < nf; i++) if (i != gfam && (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { cfam = i; break; }
    if (cfam == UINT32_MAX) { fprintf(stderr, "NO_MQ: no second queue family with COMPUTE (%u families)\n", nf); exit(3); }
  } else cfam = gfam;
  float pr = 1.0f;
  VkDeviceQueueCreateInfo qci[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
  qci[0].queueFamilyIndex = gfam; qci[0].queueCount = 1; qci[0].pQueuePriorities = &pr;
  qci[1].queueFamilyIndex = cfam; qci[1].queueCount = 1; qci[1].pQueuePriorities = &pr;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.timelineSemaphore = VK_TRUE;
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.synchronization2 = VK_TRUE; f13.dynamicRendering = VK_TRUE; f13.pNext = &f12;
  const char* dext[] = {"VK_KHR_swapchain"};
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &f13; dci.queueCreateInfoCount = mq ? 2 : 1; dci.pQueueCreateInfos = qci;
  dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = dext;
  CK(vkCreateDevice(pd, &dci, nullptr, &dev));
  vkGetDeviceQueue(dev, gfam, 0, &gq);
  if (mq) vkGetDeviceQueue(dev, cfam, 0, &cq); else cq = gq;
  pSetName = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(inst, "vkSetDebugUtilsObjectNameEXT");
  NAME(VK_OBJECT_TYPE_QUEUE, gq, "gfxq");
  if (mq) NAME(VK_OBJECT_TYPE_QUEUE, cq, "compq");
  // headless surface + swapchain
  if (xcb) {
    xcb_connection_t* conn = xcb_connect(nullptr, nullptr);
    if (xcb_connection_has_error(conn)) { fprintf(stderr, "FATAL no X display\n"); exit(2); }
    xcb_screen_t* scr = xcb_setup_roots_iterator(xcb_get_setup(conn)).data;
    xcb_window_t win = xcb_generate_id(conn);
    xcb_create_window(conn, XCB_COPY_FROM_PARENT, win, scr->root, 0, 0, 64, 64, 0, XCB_WINDOW_CLASS_INPUT_OUTPUT,
                      scr->root_visual, 0, nullptr);
    xcb_map_window(conn, win); xcb_flush(conn);
    VkXcbSurfaceCreateInfoKHR xci{VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR};
    xci.connection = conn; xci.window = win;
    auto pCreateXcb = (PFN_vkCreateXcbSurfaceKHR)vkGetInstanceProcAddr(inst, "vkCreateXcbSurfaceKHR");
    CK(pCreateXcb(inst, &xci, nullptr, &surf));
  } else {
    VkHeadlessSurfaceCreateInfoEXT hci{VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT};
    auto pCreateHeadless = (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(inst, "vkCreateHeadlessSurfaceEXT");
    CK(pCreateHeadless(inst, &hci, nullptr, &surf));
  }
  VkBool32 sup = VK_FALSE; vkGetPhysicalDeviceSurfaceSupportKHR(pd, gfam, surf, &sup);
  VkSurfaceCapabilitiesKHR caps; CK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surf, &caps));
  uint32_t nfmt = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surf, &nfmt, nullptr);
  std::vector<VkSurfaceFormatKHR> fmts(nfmt); vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surf, &nfmt, fmts.data());
  sc_fmt = fmts[0].format;
  for (auto& f : fmts) if (f.format == VK_FORMAT_B8G8R8A8_UNORM) sc_fmt = f.format;
  sc_ext = caps.currentExtent.width == 0xFFFFFFFFu ? VkExtent2D{64, 64} : caps.currentExtent;
  if (sc_ext.width == 0 || sc_ext.width > 4096) sc_ext = {64, 64};
  uint32_t want_env = getenv("WF_IMAGES") ? (uint32_t)atoi(getenv("WF_IMAGES")) : 3;
  uint32_t want = caps.minImageCount < want_env ? want_env : caps.minImageCount;
  if (caps.maxImageCount && want > caps.maxImageCount) want = caps.maxImageCount;
  VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sci.surface = surf; sci.minImageCount = want; sci.imageFormat = sc_fmt; sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  sci.imageExtent = sc_ext; sci.imageArrayLayers = 1;
  sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; sci.preTransform = caps.currentTransform;
  sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; sci.presentMode = VK_PRESENT_MODE_FIFO_KHR; sci.clipped = VK_TRUE;
  CK(vkCreateSwapchainKHR(dev, &sci, nullptr, &sc));
  uint32_t ni = 0; vkGetSwapchainImagesKHR(dev, sc, &ni, nullptr);
  sc_img.resize(ni); vkGetSwapchainImagesKHR(dev, sc, &ni, sc_img.data());
  for (uint32_t i = 0; i < ni; i++) {
    NAME(VK_OBJECT_TYPE_IMAGE, sc_img[i], "swapchain[" + std::to_string(i) + "]");
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = sc_img[i]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = sc_fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v; CK(vkCreateImageView(dev, &vi, nullptr, &v)); sc_view.push_back(v);
    NAME(VK_OBJECT_TYPE_IMAGE_VIEW, v, "swapchain[" + std::to_string(i) + "].view");
  }
  fprintf(stderr, "SWAPCHAIN images=%u min=%u extent=%ux%u families=%u gfam=%u cfam=%u\n", ni, caps.minImageCount,
          sc_ext.width, sc_ext.height, nf, gfam, cfam);
}

// ---------------------------------------------------------------- small helpers
static VkSemaphore mksem(const std::string& n, bool timeline = false) {
  VkSemaphoreTypeCreateInfo t{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  t.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE; t.initialValue = 0;
  VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  if (timeline) ci.pNext = &t;
  VkSemaphore s; CK(vkCreateSemaphore(dev, &ci, nullptr, &s));
  NAME(VK_OBJECT_TYPE_SEMAPHORE, s, n);
  return s;
}
static VkFence mkfence(const std::string& n) {
  VkFenceCreateInfo ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkFence f; CK(vkCreateFence(dev, &ci, nullptr, &f));
  NAME(VK_OBJECT_TYPE_FENCE, f, n);
  return f;
}
static VkCommandPool mkpool(uint32_t fam) {
  VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; ci.queueFamilyIndex = fam;
  VkCommandPool p; CK(vkCreateCommandPool(dev, &ci, nullptr, &p)); return p;
}
static VkCommandBuffer mkcb(VkCommandPool p, const std::string& n) {
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = p; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
  VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &ai, &cb));
  NAME(VK_OBJECT_TYPE_COMMAND_BUFFER, cb, n);
  return cb;
}
static void begin(VkCommandBuffer cb) {
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  CK(vkBeginCommandBuffer(cb, &bi));
}
struct SetLayoutB { VkDescriptorType t; VkShaderStageFlags st; };
static VkDescriptorSetLayout mksetlayout(std::vector<SetLayoutB> bs) {
  std::vector<VkDescriptorSetLayoutBinding> v;
  for (uint32_t i = 0; i < bs.size(); i++) v.push_back({i, bs[i].t, 1, bs[i].st, nullptr});
  VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  ci.bindingCount = (uint32_t)v.size(); ci.pBindings = v.data();
  VkDescriptorSetLayout l; CK(vkCreateDescriptorSetLayout(dev, &ci, nullptr, &l)); return l;
}
static VkPipelineLayout mkplayout(VkDescriptorSetLayout l) {
  VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  ci.setLayoutCount = 1; ci.pSetLayouts = &l;
  VkPipelineLayout p; CK(vkCreatePipelineLayout(dev, &ci, nullptr, &p)); return p;
}
static VkDescriptorPool g_dpool;
static VkDescriptorSet mkset(VkDescriptorSetLayout l, std::vector<std::pair<VkDescriptorType, VkBuffer>> bufs, const std::string& n) {
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = g_dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &l;
  VkDescriptorSet s; CK(vkAllocateDescriptorSets(dev, &ai, &s));
  std::vector<VkDescriptorBufferInfo> bi(bufs.size());
  std::vector<VkWriteDescriptorSet> w(bufs.size());
  for (uint32_t i = 0; i < bufs.size(); i++) {
    bi[i] = {bufs[i].second, 0, VK_WHOLE_SIZE};
    w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[i].dstSet = s; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = bufs[i].first; w[i].pBufferInfo = &bi[i];
  }
  vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
  NAME(VK_OBJECT_TYPE_DESCRIPTOR_SET, s, n);
  return s;
}
static VkPipeline mkcompute(const char* spv, VkPipelineLayout pl) {
  VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = shader(spv); ci.stage.pName = "main";
  ci.layout = pl;
  VkPipeline p; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p)); return p;
}
static VkPipeline mkgraphics(const char* vs, const char* fs, VkPipelineLayout pl, VkFormat color, VkFormat depth, bool vinput) {
  VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
  st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = shader(vs); st[0].pName = "main";
  st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = shader(fs); st[1].pName = "main";
  VkVertexInputBindingDescription vb{0, 16, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription va{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0};
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  if (vinput) { vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb; vi.vertexAttributeDescriptionCount = 1; vi.pVertexAttributeDescriptions = &va; }
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = depth != VK_FORMAT_UNDEFINED; ds.depthWriteEnable = ds.depthTestEnable; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1; cb.pAttachments = &cba;
  VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  ri.colorAttachmentCount = 1; ri.pColorAttachmentFormats = &color; ri.depthAttachmentFormat = depth;
  VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  ci.pNext = &ri; ci.stageCount = 2; ci.pStages = st; ci.pVertexInputState = &vi; ci.pInputAssemblyState = &ia;
  ci.pViewportState = &vp; ci.pRasterizationState = &rs; ci.pMultisampleState = &ms; ci.pDepthStencilState = &ds;
  ci.pColorBlendState = &cb; ci.pDynamicState = &dy; ci.layout = pl;
  VkPipeline p; CK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p)); return p;
}
static void viewport(VkCommandBuffer cb, VkExtent2D e) {
  VkViewport v{0, 0, (float)e.width, (float)e.height, 0, 1}; VkRect2D s{{0, 0}, e};
  vkCmdSetViewport(cb, 0, 1, &v); vkCmdSetScissor(cb, 0, 1, &s);
}
static void begin_rendering(VkCommandBuffer cb, VkImageView color, VkExtent2D e, VkImageView depth = VK_NULL_HANDLE) {
  VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  ca.imageView = color; ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  ca.clearValue.color = {{0.1f, 0.1f, 0.2f, 1.0f}};
  VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  da.imageView = depth; da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  da.clearValue.depthStencil = {1.0f, 0};
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, e}; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &ca;
  if (depth) ri.pDepthAttachment = &da;
  vkCmdBeginRendering(cb, &ri);
}

// frame-level semaphore plumbing shared by the workloads
struct FrameSync {
  int F, nimg;
  std::vector<VkSemaphore> acq, rend;   // acquire per slot; render-finished per image
  std::vector<VkFence> fence;
  void init(int f, int n, bool fences) {
    F = f; nimg = n;
    for (int i = 0; i < F; i++) acq.push_back(mksem("acquire_sem" + std::to_string(i)));
    for (int i = 0; i < n; i++) rend.push_back(mksem("render_sem" + std::to_string(i)));
    if (fences) for (int i = 0; i < F; i++) fence.push_back(mkfence("frame_fence" + std::to_string(i)));
  }
  VkSemaphore acq_sem(int slot) { return mut("acq_sem") == "shared" ? acq[0] : acq[slot]; }
  VkSemaphore rend_sem(int slot, uint32_t img) {
    std::string k = mut("render_sem");
    if (k == "shared") return rend[0];
    if (k == "per_frame") return rend[slot % rend.size()];
    return rend[img];
  }
  // returns acquire-wait info; ok=false when dropped
  bool acq_wait(VkSemaphore s, VkPipelineStageFlags2 stage, VkPipelineStageFlags2 wrong, VkPipelineStageFlags2 early,
                VkSemaphoreSubmitInfo& out) {
    out = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    out.semaphore = s; out.stageMask = stage;
    std::string k = mut("acq_wait");
    if (k == "drop") return false;
    if (k == "stage_none") out.stageMask = VK_PIPELINE_STAGE_2_NONE;
    if (k == "stage_wrong") out.stageMask = wrong;
    if (k == "stage_early" && early) out.stageMask = early;
    return true;
  }
  VkSemaphoreSubmitInfo signal(VkSemaphore s, VkPipelineStageFlags2 wrong, VkPipelineStageFlags2 late) {
    VkSemaphoreSubmitInfo o{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    o.semaphore = s; o.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    std::string k = mut("signal_stage");
    if (k == "none") o.stageMask = VK_PIPELINE_STAGE_2_NONE;
    if (k == "wrong") o.stageMask = wrong;
    if (k == "late") o.stageMask = late;   // the last stage that writes the image (common in real code)
    // manual experiment only (not a catalogue mutation): force a stage mask, e.g. ALL_GRAPHICS = 0x8000
    if (const char* e = getenv("WF_SIGNAL_STAGE_HEX")) o.stageMask = strtoull(e, nullptr, 16);
    return o;
  }
  void present(VkSemaphore wait, uint32_t img) {
    if (mut("present") == "skip") return;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    if (mut("present_wait") != "drop") { pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &wait; }
    pi.swapchainCount = 1; pi.pSwapchains = &sc; pi.pImageIndices = &img;
    CK(vkQueuePresentKHR(gq, &pi));
  }
  void fence_wait(int slot) {
    std::string k = mut("fence_wait");
    if (k != "drop") {
      VkFence f = fence[k == "other_slot" ? (slot + 1) % F : slot];
      CK(vkWaitForFences(dev, 1, &f, VK_TRUE, UINT64_MAX));
    }
    CK(vkResetFences(dev, 1, &fence[slot]));
  }
};
static uint32_t acquire(VkSemaphore s) {
  uint32_t idx = 0;
  CK(vkAcquireNextImageKHR(dev, sc, UINT64_MAX, s, VK_NULL_HANDLE, &idx));
  fprintf(stderr, "ACQ %d %u\n", g_frame, idx);
  return idx;
}
static void submit(VkQueue q, VkCommandBuffer cb, std::vector<VkSemaphoreSubmitInfo> waits,
                   std::vector<VkSemaphoreSubmitInfo> sigs, VkFence f) {
  VkCommandBufferSubmitInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}; ci.commandBuffer = cb;
  VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  si.waitSemaphoreInfoCount = (uint32_t)waits.size(); si.pWaitSemaphoreInfos = waits.data();
  si.commandBufferInfoCount = 1; si.pCommandBufferInfos = &ci;
  si.signalSemaphoreInfoCount = (uint32_t)sigs.size(); si.pSignalSemaphoreInfos = sigs.data();
  CK(vkQueueSubmit2(q, 1, &si, f));
}
static int pick(const char* site, int slot) { return mut(site) == "shared" ? 0 : slot; }

// ---------------------------------------------------------------- W1: offscreen
static void run_offscreen(int F, int N) {
  const VkExtent2D oe{64, 64};
  const uint32_t NP = 64;
  VkCommandPool pool = mkpool(gfam);
  FrameSync fs; fs.init(F, (int)sc_img.size(), true);
  std::vector<Buf> ubo; std::vector<Img> off; std::vector<VkCommandBuffer> cbs;
  for (int i = 0; i < F; i++) {
    ubo.push_back(mkbuf("ubo" + std::to_string(i), 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true));
    off.push_back(mkimg("offscreen" + std::to_string(i), VK_FORMAT_R8G8B8A8_UNORM, oe,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT));
    cbs.push_back(mkcb(pool, "cb" + std::to_string(i)));
  }
  Buf sbuf = mkbuf("particles", NP * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  auto cl = mksetlayout({{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto gl = mksetlayout({{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto cpl = mkplayout(cl), gpl = mkplayout(gl);
  VkPipeline cp = mkcompute("sim1.comp.spv", cpl);
  VkPipeline gp = mkgraphics("draw.vert.spv", "flat.frag.spv", gpl, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_UNDEFINED, false);
  std::vector<VkDescriptorSet> cset, gset;
  for (int i = 0; i < F; i++) {
    cset.push_back(mkset(cl, {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo[i].b}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sbuf.b}}, "cset" + std::to_string(i)));
    gset.push_back(mkset(gl, {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo[i].b}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sbuf.b}}, "gset" + std::to_string(i)));
  }
  // barrier sites (correct masks, then the wrong alternative)
  const Site B_SBUF_WAR{"b_sbuf_war", S2(VERTEX_SHADER), NONE_A, S2(COMPUTE_SHADER), A2(SHADER_STORAGE_WRITE),
                        S2(FRAGMENT_SHADER), NONE_A, S2(VERTEX_SHADER), A2(SHADER_STORAGE_WRITE), false};
  const Site B_SBUF_RAW{"b_sbuf_raw", S2(COMPUTE_SHADER), A2(SHADER_STORAGE_WRITE), S2(VERTEX_SHADER), A2(SHADER_STORAGE_READ),
                        S2(VERTEX_SHADER), A2(SHADER_STORAGE_WRITE), S2(FRAGMENT_SHADER), A2(SHADER_STORAGE_READ), false};
  const Site B_OFF_COLOR{"b_off_color", S2(ALL_TRANSFER), NONE_A, S2(COLOR_ATTACHMENT_OUTPUT), A2(COLOR_ATTACHMENT_WRITE),
                         S2(FRAGMENT_SHADER), NONE_A, S2(FRAGMENT_SHADER), A2(SHADER_SAMPLED_READ), true};
  const Site B_OFF_SRC{"b_off_src", S2(COLOR_ATTACHMENT_OUTPUT), A2(COLOR_ATTACHMENT_WRITE), S2(BLIT), A2(TRANSFER_READ),
                       S2(FRAGMENT_SHADER), A2(SHADER_STORAGE_WRITE), S2(COPY), A2(TRANSFER_READ), true};
  const Site B_SC_DST{"b_sc_dst", S2(BLIT), NONE_A, S2(BLIT), A2(TRANSFER_WRITE),
                      S2(COLOR_ATTACHMENT_OUTPUT), NONE_A, S2(COPY), A2(TRANSFER_WRITE), true};
  const Site B_SC_PRESENT{"b_sc_present", S2(BLIT), A2(TRANSFER_WRITE), NONE_S, NONE_A,
                          S2(COPY), A2(TRANSFER_WRITE), S2(BLIT), A2(TRANSFER_READ), true};
  for (int f = 0; f < N; f++) {
    g_frame = f;
    int slot = f % F;
    fs.fence_wait(slot);
    int us = pick("ubo", slot), os = pick("off", slot), cs = pick("cb", slot);
    float* u = (float*)ubo[us].map; u[0] = 0.1f * (f + 1); u[1] = (float)f; u[2] = 1.0f + f; u[3] = 2.0f * f + 0.5f;
    VkSemaphore as = fs.acq_sem(slot);
    uint32_t img = acquire(as);
    VkCommandBuffer cb = cbs[cs];
    begin(cb);
    bar_buf(cb, B_SBUF_WAR, sbuf.b);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cpl, 0, 1, &cset[us], 0, nullptr);
    vkCmdDispatch(cb, NP / 64, 1, 1);
    bar_buf(cb, B_SBUF_RAW, sbuf.b);
    bar_img(cb, B_OFF_COLOR, off[os].i, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    begin_rendering(cb, off[os].v, oe);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, gp);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, gpl, 0, 1, &gset[us], 0, nullptr);
    viewport(cb, oe);
    vkCmdDraw(cb, NP * 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    bar_img(cb, B_OFF_SRC, off[os].i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    bar_img(cb, B_SC_DST, sc_img[img], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageBlit bl{};
    bl.srcSubresource = bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    bl.srcOffsets[1] = {(int)oe.width, (int)oe.height, 1};
    bl.dstOffsets[1] = {(int)sc_ext.width, (int)sc_ext.height, 1};
    vkCmdBlitImage(cb, off[os].i, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc_img[img], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_NEAREST);
    bar_img(cb, B_SC_PRESENT, sc_img[img], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    CK(vkEndCommandBuffer(cb));
    std::vector<VkSemaphoreSubmitInfo> waits;
    VkSemaphoreSubmitInfo w;
    if (fs.acq_wait(as, S2(BLIT), S2(COLOR_ATTACHMENT_OUTPUT), 0, w)) waits.push_back(w);
    VkSemaphore rs = fs.rend_sem(slot, img);
    submit(gq, cb, waits, {fs.signal(rs, S2(COMPUTE_SHADER), S2(BLIT))}, fs.fence[slot]);
    fs.present(rs, img);
  }
  vkDeviceWaitIdle(dev);
}

// ---------------------------------------------------------------- W2: timeline (+ multi-queue)
static void run_timeline(int F, int N) {
  const uint32_t NP = 64;
  VkCommandPool gpool = mkpool(gfam), cpool = mkpool(cfam);
  FrameSync fs; fs.init(F, (int)sc_img.size(), false);
  VkSemaphore tlc = mksem("tl_compute", true), tlg = mksem("tl_graphics", true);
  std::vector<Buf> params, sim; std::vector<VkCommandBuffer> ccb, gcb;
  for (int i = 0; i < F; i++) {
    params.push_back(mkbuf("params" + std::to_string(i), 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true));
    sim.push_back(mkbuf("sim" + std::to_string(i), NP * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, !getenv("WF_CONCURRENT")));
    ccb.push_back(mkcb(cpool, "ccb" + std::to_string(i)));
    gcb.push_back(mkcb(gpool, "gcb" + std::to_string(i)));
  }
  Buf state = mkbuf("state", NP * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, true);
  auto cl = mksetlayout({{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}});
  auto gl = mksetlayout({{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto cpl = mkplayout(cl), gpl = mkplayout(gl);
  VkPipeline cp = mkcompute("sim2.comp.spv", cpl);
  VkPipeline gp = mkgraphics("sim.vert.spv", "flat.frag.spv", gpl, sc_fmt, VK_FORMAT_UNDEFINED, false);
  // sets for every (params, sim) pair so the index mutations need no descriptor updates
  std::vector<VkDescriptorSet> cset(F * F), gset(F);
  for (int p = 0; p < F; p++)
    for (int s = 0; s < F; s++)
      cset[p * F + s] = mkset(cl, {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, params[p].b}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sim[s].b},
                                   {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, state.b}}, "cset" + std::to_string(p) + std::to_string(s));
  for (int s = 0; s < F; s++) gset[s] = mkset(gl, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sim[s].b}}, "gset" + std::to_string(s));
  const Site B_STATE{"b_state", S2(COMPUTE_SHADER), A2(SHADER_STORAGE_WRITE), S2(COMPUTE_SHADER), A2(SHADER_STORAGE_READ) | A2(SHADER_STORAGE_WRITE),
                     S2(VERTEX_SHADER), A2(SHADER_STORAGE_WRITE), S2(VERTEX_SHADER), A2(SHADER_STORAGE_READ), false};
  const Site B_SC_COLOR{"b_sc_color", S2(COLOR_ATTACHMENT_OUTPUT), NONE_A, S2(COLOR_ATTACHMENT_OUTPUT), A2(COLOR_ATTACHMENT_WRITE),
                        S2(FRAGMENT_SHADER), NONE_A, S2(FRAGMENT_SHADER), A2(SHADER_SAMPLED_READ), true};
  const Site B_SC_PRESENT{"b_sc_present", S2(COLOR_ATTACHMENT_OUTPUT), A2(COLOR_ATTACHMENT_WRITE), NONE_S, NONE_A,
                          S2(FRAGMENT_SHADER), A2(SHADER_STORAGE_WRITE), S2(COLOR_ATTACHMENT_OUTPUT), A2(COLOR_ATTACHMENT_READ), true};
  const Site Q_REL{"qfot_rel", S2(COMPUTE_SHADER), A2(SHADER_STORAGE_WRITE), NONE_S, NONE_A, 0, 0, 0, 0, true};
  const Site Q_ACQ{"qfot_acq", NONE_S, NONE_A, S2(VERTEX_SHADER), A2(SHADER_STORAGE_READ),
                   0, 0, S2(FRAGMENT_SHADER), A2(SHADER_STORAGE_READ), true};
  bool qfot = mq && gfam != cfam && !getenv("WF_CONCURRENT");
  for (int f = 0; f < N; f++) {
    g_frame = f;
    int slot = f % F;
    uint64_t v = f + 1;
    // host frame pacing: graphics of frame f-F must be done before slot is reused
    std::string hk = mut("tl_host_wait");
    if (v > (uint64_t)F && hk != "drop") {
      uint64_t want = v - F - (hk == "value_prev" ? 1 : 0);
      if (want > 0) {
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1; wi.pSemaphores = &tlg; wi.pValues = &want;
        CK(vkWaitSemaphores(dev, &wi, UINT64_MAX));
      }
    }
    int ps = pick("params", slot), ss = pick("sim", slot), cs = pick("cb", slot);
    float* u = (float*)params[ps].map; u[0] = 0.1f * (f + 1); u[1] = (float)f; u[2] = 3.0f + f; u[3] = 0.25f * f + 1.0f;
    // compute submit
    VkCommandBuffer c = ccb[cs];
    begin(c);
    bar_buf(c, B_STATE, state.b);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, cp);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, cpl, 0, 1, &cset[ps * F + ss], 0, nullptr);
    vkCmdDispatch(c, NP / 64, 1, 1);
    if (qfot) bar_buf(c, Q_REL, sim[ss].b, cfam, gfam);
    CK(vkEndCommandBuffer(c));
    VkSemaphoreSubmitInfo sig{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    sig.semaphore = tlc; sig.value = v + (mut("tl_comp_signal") == "value_next" ? 1 : 0);
    sig.stageMask = S2(ALL_COMMANDS);
    // GPU-side WAR protection: compute of frame f reuses sim[slot] / params[slot] that graphics of
    // frame f-F read, so it waits for tlg >= f+1-F (the async-compute pattern)
    std::vector<VkSemaphoreSubmitInfo> cwaits;
    std::string ck = mut("tl_comp_wait");
    if (v > (uint64_t)F && ck != "drop") {
      VkSemaphoreSubmitInfo cw{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      cw.semaphore = tlg; cw.value = v - F - (ck == "value_prev" ? 1 : 0); cw.stageMask = S2(COMPUTE_SHADER);
      if (ck == "stage_wrong") cw.stageMask = S2(ALL_TRANSFER);
      if (cw.value > 0) cwaits.push_back(cw);
    }
    submit(cq, c, cwaits, {sig}, VK_NULL_HANDLE);
    // graphics submit
    VkSemaphore as = fs.acq_sem(slot);
    uint32_t img = acquire(as);
    VkCommandBuffer g = gcb[cs];
    begin(g);
    if (qfot) bar_buf(g, Q_ACQ, sim[ss].b, cfam, gfam);
    bar_img(g, B_SC_COLOR, sc_img[img], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    begin_rendering(g, sc_view[img], sc_ext);
    vkCmdBindPipeline(g, VK_PIPELINE_BIND_POINT_GRAPHICS, gp);
    vkCmdBindDescriptorSets(g, VK_PIPELINE_BIND_POINT_GRAPHICS, gpl, 0, 1, &gset[ss], 0, nullptr);
    viewport(g, sc_ext);
    vkCmdDraw(g, NP * 3, 1, 0, 0);
    vkCmdEndRendering(g);
    bar_img(g, B_SC_PRESENT, sc_img[img], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    CK(vkEndCommandBuffer(g));
    std::vector<VkSemaphoreSubmitInfo> waits;
    std::string gk = mut("tl_gfx_wait");
    if (gk != "drop") {
      VkSemaphoreSubmitInfo tw{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
      tw.semaphore = tlc; tw.value = v - (gk == "value_prev" ? 1 : 0); tw.stageMask = S2(VERTEX_SHADER);
      if (gk == "stage_wrong") tw.stageMask = S2(FRAGMENT_SHADER);
      if (gk == "stage_early") tw.stageMask = S2(VERTEX_ATTRIBUTE_INPUT);
      if (gk == "stage_all") tw.stageMask = S2(ALL_COMMANDS);
      if (tw.value > 0) waits.push_back(tw);
    }
    VkSemaphoreSubmitInfo w;
    if (fs.acq_wait(as, S2(COLOR_ATTACHMENT_OUTPUT), S2(ALL_TRANSFER), S2(VERTEX_SHADER), w)) waits.push_back(w);
    VkSemaphore rs = fs.rend_sem(slot, img);
    VkSemaphoreSubmitInfo tg{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    tg.semaphore = tlg; tg.value = v; tg.stageMask = S2(ALL_COMMANDS);
    submit(gq, g, waits, {fs.signal(rs, S2(VERTEX_SHADER), S2(COLOR_ATTACHMENT_OUTPUT)), tg}, VK_NULL_HANDLE);
    fs.present(rs, img);
  }
  vkDeviceWaitIdle(dev);
}

// ---------------------------------------------------------------- W3: upload
static void run_upload(int F, int N) {
  const uint32_t NV = 48;                 // 16 triangles
  const VkDeviceSize REG = 4096;          // one page per ring region
  VkCommandPool pool = mkpool(gfam);
  FrameSync fs; fs.init(F, (int)sc_img.size(), true);
  Buf ring = mkbuf("staging_ring", REG * F, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
  Buf vbuf = mkbuf("vertices", NV * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false);
  Img depth = mkimg("depth", VK_FORMAT_D32_SFLOAT, sc_ext, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
  std::vector<VkCommandBuffer> cbs;
  for (int i = 0; i < F; i++) cbs.push_back(mkcb(pool, "cb" + std::to_string(i)));
  VkPipelineLayoutCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pci, nullptr, &pl));
  VkPipeline gp = mkgraphics("vbuf.vert.spv", "flat.frag.spv", pl, sc_fmt, VK_FORMAT_D32_SFLOAT, true);
  const Site B_VBUF_WAR{"b_vbuf_war", S2(VERTEX_ATTRIBUTE_INPUT), NONE_A, S2(COPY), A2(TRANSFER_WRITE),
                        S2(FRAGMENT_SHADER), NONE_A, S2(BLIT), A2(TRANSFER_WRITE), false};
  const Site B_VBUF_RAW{"b_vbuf_raw", S2(COPY), A2(TRANSFER_WRITE), S2(VERTEX_ATTRIBUTE_INPUT), A2(VERTEX_ATTRIBUTE_READ),
                        S2(BLIT), A2(TRANSFER_WRITE), S2(VERTEX_SHADER), A2(SHADER_READ), false};
  const Site B_DEPTH{"b_depth", S2(LATE_FRAGMENT_TESTS), A2(DEPTH_STENCIL_ATTACHMENT_WRITE),
                     S2(EARLY_FRAGMENT_TESTS) | S2(LATE_FRAGMENT_TESTS), A2(DEPTH_STENCIL_ATTACHMENT_READ) | A2(DEPTH_STENCIL_ATTACHMENT_WRITE),
                     S2(EARLY_FRAGMENT_TESTS), A2(DEPTH_STENCIL_ATTACHMENT_WRITE), S2(LATE_FRAGMENT_TESTS), A2(DEPTH_STENCIL_ATTACHMENT_WRITE), true};
  const Site B_SC_COLOR{"b_sc_color", S2(COLOR_ATTACHMENT_OUTPUT), NONE_A, S2(COLOR_ATTACHMENT_OUTPUT), A2(COLOR_ATTACHMENT_WRITE),
                        S2(FRAGMENT_SHADER), NONE_A, S2(FRAGMENT_SHADER), A2(SHADER_SAMPLED_READ), true};
  const Site B_SC_PRESENT{"b_sc_present", S2(COLOR_ATTACHMENT_OUTPUT), A2(COLOR_ATTACHMENT_WRITE), NONE_S, NONE_A,
                          S2(FRAGMENT_SHADER), A2(SHADER_STORAGE_WRITE), S2(COLOR_ATTACHMENT_OUTPUT), A2(COLOR_ATTACHMENT_READ), true};
  for (int f = 0; f < N; f++) {
    g_frame = f;
    int slot = f % F;
    fs.fence_wait(slot);
    int rs_ = pick("ring", slot), cs = pick("cb", slot);
    float* v = (float*)((char*)ring.map + REG * rs_);
    for (uint32_t i = 0; i < NV; i++) {
      float a = 0.37f * i + 0.05f * f;
      v[4 * i + 0] = 0.8f * __builtin_cosf(a); v[4 * i + 1] = 0.8f * __builtin_sinf(a);
      v[4 * i + 2] = 0.01f * (i % 7) + 0.001f * f; v[4 * i + 3] = 1.0f + f;
    }
    VkSemaphore as = fs.acq_sem(slot);
    uint32_t img = acquire(as);
    VkCommandBuffer cb = cbs[cs];
    begin(cb);
    bar_buf(cb, B_VBUF_WAR, vbuf.b);
    VkBufferCopy bc{REG * rs_, 0, NV * 16};
    vkCmdCopyBuffer(cb, ring.b, vbuf.b, 1, &bc);
    bar_buf(cb, B_VBUF_RAW, vbuf.b);
    bar_img(cb, B_DEPTH, depth.i, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    bar_img(cb, B_SC_COLOR, sc_img[img], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    begin_rendering(cb, sc_view[img], sc_ext, depth.v);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, gp);
    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &vbuf.b, &zero);
    viewport(cb, sc_ext);
    vkCmdDraw(cb, NV, 1, 0, 0);
    vkCmdEndRendering(cb);
    bar_img(cb, B_SC_PRESENT, sc_img[img], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    CK(vkEndCommandBuffer(cb));
    std::vector<VkSemaphoreSubmitInfo> waits;
    VkSemaphoreSubmitInfo w;
    if (fs.acq_wait(as, S2(COLOR_ATTACHMENT_OUTPUT), S2(ALL_TRANSFER), S2(VERTEX_SHADER), w)) waits.push_back(w);
    VkSemaphore rsem = fs.rend_sem(slot, img);
    submit(gq, cb, waits, {fs.signal(rsem, S2(VERTEX_SHADER), S2(COLOR_ATTACHMENT_OUTPUT))}, fs.fence[slot]);
    fs.present(rsem, img);
  }
  vkDeviceWaitIdle(dev);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: wf offscreen|timeline|upload [--fif N] [--frames N] [--mq] [--sites]\n"); return 1; }
  std::string w = argv[1];
  int F = 2, N = 6; bool sites = false;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--fif" && i + 1 < argc) F = atoi(argv[++i]);
    else if (a == "--frames" && i + 1 < argc) N = atoi(argv[++i]);
    else if (a == "--mq") mq = true;
    else if (a == "--sites") sites = true;
  }
  if (sites) {
    printf("[");
    auto c = catalogue(w);
    for (size_t i = 0; i < c.size(); i++) {
      printf("%s{\"site\":\"%s\",\"what\":\"%s\",\"kinds\":[", i ? "," : "", c[i].site, c[i].what);
      for (size_t k = 0; k < c[i].kinds.size(); k++) printf("%s\"%s\"", k ? "," : "", c[i].kinds[k]);
      printf("]}");
    }
    printf("]\n");
    return 0;
  }
  if (const char* s = getenv("WF_SHADERS")) g_shaders = s;
  parse_muts();
  init_device();
  VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64}};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 64; dpi.poolSizeCount = 2; dpi.pPoolSizes = ps;
  CK(vkCreateDescriptorPool(dev, &dpi, nullptr, &g_dpool));
  if (w == "offscreen") run_offscreen(F, N);
  else if (w == "timeline") run_timeline(F, N);
  else if (w == "upload") run_upload(F, N);
  else { fprintf(stderr, "unknown workload %s\n", w.c_str()); return 1; }
  fprintf(stderr, "WF-DONE\n");
  fflush(stderr);
  exit(0);   // no teardown: object destruction is not under test
}
