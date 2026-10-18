// VK_LAYER_COVENANT_probe -- measurement helper for running UNMODIFIED sample
// programs with and without a COVENANT barrier-elision plan.
//
// It never changes what the application records. It adds, below the
// application and below VK_LAYER_COVENANT_elide (and below the Khronos
// validation layer when that is enabled, so validation never sees it):
//
//   COVENANT_PROBE_HASH=<file>        after the application's work for a present is
//                                     done, copy each presented swapchain image to a
//                                     host buffer and append one line
//                                       frame <n> swapchain <k> image <i> <w>x<h> format <f> fnv1a64 <hash>
//                                     (adds TRANSFER_SRC to the swapchain's usage)
//   COVENANT_PROBE_HASH_FRAMES=a,b,.. hash only these 0-based present indices
//                                     (default: every present)
//   COVENANT_PROBE_TIMESTAMPS=<file>  wrap every submitted batch in two command buffers
//                                     that write a TOP_OF_PIPE timestamp before the
//                                     batch and a BOTTOM_OF_PIPE timestamp after it, and
//                                     write per-batch results as JSON at exit (or at
//                                     vkDestroyDevice)
//   COVENANT_PROBE_DUMP=<dir>         also write every hashed image as <dir>/frame<n>_<k>.ppm
//                                     (8-bit 4-byte formats only; for eyeballing)
//   COVENANT_PROBE_QUIET=1            no summary line on stderr
//
// It also calls covenant_fakeclock_tick() at every present when that symbol is
// in the process (bench/elide/real/fakeclock.cpp, Linux LD_PRELOAD shim): that
// makes the samples' animation time advance by exactly 1/60 s per frame, so the
// hashes of two runs are comparable.
//
// Layer order (VK_INSTANCE_LAYERS, first = closest to the application):
//   VK_LAYER_COVENANT_elide:VK_LAYER_KHRONOS_validation:VK_LAYER_COVENANT_probe
#include <vulkan/vulkan.h>
#if defined(__has_include)
#if __has_include(<vulkan/vk_layer.h>)
#include <vulkan/vk_layer.h>
#define HAVE_VK_LAYER_H 1
#endif
#endif

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#define EXPORT extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

#ifndef HAVE_VK_LAYER_H
typedef enum VkLayerFunction_ { VK_LAYER_LINK_INFO = 0, VK_LOADER_DATA_CALLBACK = 1 } VkLayerFunction;
typedef PFN_vkVoidFunction(VKAPI_PTR* PFN_GetPhysicalDeviceProcAddr)(VkInstance, const char*);
typedef VkResult(VKAPI_PTR* PFN_vkSetDeviceLoaderData)(VkDevice device, void* object);
typedef struct VkLayerInstanceLink_ {
  struct VkLayerInstanceLink_* pNext;
  PFN_vkGetInstanceProcAddr pfnNextGetInstanceProcAddr;
  PFN_GetPhysicalDeviceProcAddr pfnNextGetPhysicalDeviceProcAddr;
} VkLayerInstanceLink;
typedef struct {
  VkStructureType sType;
  const void* pNext;
  VkLayerFunction function;
  union { VkLayerInstanceLink* pLayerInfo; void* other; } u;
} VkLayerInstanceCreateInfo;
typedef struct VkLayerDeviceLink_ {
  struct VkLayerDeviceLink_* pNext;
  PFN_vkGetInstanceProcAddr pfnNextGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr pfnNextGetDeviceProcAddr;
} VkLayerDeviceLink;
typedef struct {
  VkStructureType sType;
  const void* pNext;
  VkLayerFunction function;
  union { VkLayerDeviceLink* pLayerInfo; PFN_vkSetDeviceLoaderData pfnSetDeviceLoaderData; } u;
} VkLayerDeviceCreateInfo;
typedef enum VkNegotiateLayerStructType { LAYER_NEGOTIATE_UNINTIALIZED = 0, LAYER_NEGOTIATE_INTERFACE_STRUCT = 1 } VkNegotiateLayerStructType;
typedef struct VkNegotiateLayerInterface {
  VkNegotiateLayerStructType sType;
  void* pNext;
  uint32_t loaderLayerInterfaceVersion;
  PFN_vkGetInstanceProcAddr pfnGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr pfnGetDeviceProcAddr;
  PFN_GetPhysicalDeviceProcAddr pfnGetPhysicalDeviceProcAddr;
} VkNegotiateLayerInterface;
#endif

namespace {

void* key(const void* dispatchable) { return *(void**)dispatchable; }

void vlog(const char* fmt, ...) {
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "[covenant-probe] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

std::string env(const char* n) { const char* v = getenv(n); return v ? std::string(v) : std::string(); }

#define DEV_FNS(X)                                                                                       \
  X(DestroyDevice) X(GetDeviceQueue) X(GetDeviceQueue2) X(CreateSwapchainKHR) X(DestroySwapchainKHR)   \
  X(GetSwapchainImagesKHR) X(QueuePresentKHR) X(QueueSubmit) X(QueueSubmit2) X(QueueSubmit2KHR)        \
  X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer)           \
  X(EndCommandBuffer) X(ResetCommandBuffer) X(CmdPipelineBarrier) X(CmdCopyImageToBuffer)              \
  X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory)     \
  X(BindBufferMemory) X(MapMemory) X(UnmapMemory) X(InvalidateMappedMemoryRanges) X(CreateFence)       \
  X(DestroyFence) X(WaitForFences) X(ResetFences) X(CreateSemaphore) X(DestroySemaphore)               \
  X(CreateQueryPool) X(DestroyQueryPool) X(CmdResetQueryPool) X(CmdWriteTimestamp)                     \
  X(GetQueryPoolResults) X(DeviceWaitIdle) X(QueueWaitIdle)

struct Disp {
#define DECL(n) PFN_vk##n n = nullptr;
  DEV_FNS(DECL)
#undef DECL
};

struct Swapchain {
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D extent{};
  std::vector<VkImage> images;
  uint32_t index = 0;   // creation ordinal
};

// one ring slot of the timestamp wrapper
struct Slot {
  VkCommandBuffer start = VK_NULL_HANDLE, end = VK_NULL_HANDLE;
  bool pending = false;
  uint64_t frame = 0, batch = 0;
};

struct QueueState {
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t family = 0, index = 0, ordinal = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  // hashing
  VkCommandBuffer hash_cb = VK_NULL_HANDLE;
  VkFence hash_fence = VK_NULL_HANDLE;
  VkSemaphore hash_sem = VK_NULL_HANDLE;
  // timestamps
  VkQueryPool qpool = VK_NULL_HANDLE;
  std::vector<Slot> ring;
  uint32_t next = 0;
};

struct Result { uint64_t frame, batch; uint32_t queue; uint64_t t0, t1; };

struct Device {
  VkDevice dev = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  PFN_vkGetDeviceProcAddr gdpa = nullptr;
  PFN_vkSetDeviceLoaderData set_loader_data = nullptr;
  Disp d;
  VkPhysicalDeviceMemoryProperties mem{};
  VkPhysicalDeviceProperties props{};
  std::vector<VkQueueFamilyProperties> fams;
  std::unordered_map<VkQueue, QueueState> queues;
  std::unordered_map<VkSwapchainKHR, Swapchain> swapchains;
  uint32_t swapchain_count = 0;
  // host-visible readback buffer
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory buf_mem = VK_NULL_HANDLE;
  VkDeviceSize buf_size = 0;
  void* buf_ptr = nullptr;
  bool buf_coherent = false;
  std::vector<Result> results;
  uint64_t batches = 0;
  bool flushed = false;
};

struct Instance {
  PFN_vkGetInstanceProcAddr gipa = nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties mem_props = nullptr;
  PFN_vkGetPhysicalDeviceProperties props = nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyProperties qfam = nullptr;
};

struct Global {
  std::mutex mu;
  bool loaded = false, quiet = false;
  std::string hash_path, ts_path, dump_dir;
  std::set<uint64_t> hash_frames;   // empty = all
  FILE* hash_file = nullptr;
  std::unordered_map<void*, Instance> instances;   // dispatch key -> instance
  std::unordered_map<VkPhysicalDevice, void*> phys_inst;
  std::unordered_map<void*, Device*> devices;      // dispatch key -> device
  std::atomic<uint64_t> presents{0};
  uint64_t hashed = 0;
  void (*fakeclock_tick)() = nullptr;
};
Global& G() { static Global* g = new Global(); return *g; }

constexpr uint32_t RING = 512;

void load_env() {
  Global& g = G();
  std::lock_guard<std::mutex> lk(g.mu);
  if (g.loaded) return;
  g.loaded = true;
  g.hash_path = env("COVENANT_PROBE_HASH");
  g.ts_path = env("COVENANT_PROBE_TIMESTAMPS");
  g.quiet = env("COVENANT_PROBE_QUIET") == "1";
  g.dump_dir = env("COVENANT_PROBE_DUMP");
  std::string fr = env("COVENANT_PROBE_HASH_FRAMES");
  for (size_t p = 0; p < fr.size();) {
    size_t q = fr.find(',', p);
    if (q == std::string::npos) q = fr.size();
    if (q > p) g.hash_frames.insert(strtoull(fr.substr(p, q - p).c_str(), nullptr, 10));
    p = q + 1;
  }
  if (!g.hash_path.empty()) {
    g.hash_file = fopen(g.hash_path.c_str(), "w");
    if (!g.hash_file) vlog("cannot write %s; hashing off", g.hash_path.c_str());
  }
#if defined(_WIN32)
  g.fakeclock_tick = nullptr;
#else
  g.fakeclock_tick = (void (*)())dlsym(RTLD_DEFAULT, "covenant_fakeclock_tick");
#endif
  if (!g.quiet)
    vlog("hash %s, timestamps %s%s", g.hash_file ? g.hash_path.c_str() : "off", g.ts_path.empty() ? "off" : g.ts_path.c_str(),
         g.fakeclock_tick ? ", fake clock found (1/60 s per present)" : "");
}

uint32_t bytes_per_pixel(VkFormat f) {
  switch (f) {
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_B5G6R5_UNORM_PACK16: return 2;
    default: return 4;
  }
}

uint64_t fnv1a(const uint8_t* p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; }
  return h;
}

int32_t mem_type(Device& D, uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < D.mem.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (D.mem.memoryTypes[i].propertyFlags & want) == want) return (int32_t)i;
  return -1;
}

bool alloc_cbs(Device& D, QueueState& q, uint32_t n, VkCommandBuffer* out) {
  if (!q.pool) {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = q.family;
    if (D.d.CreateCommandPool(D.dev, &ci, nullptr, &q.pool) != VK_SUCCESS) return false;
  }
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = q.pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = n;
  if (D.d.AllocateCommandBuffers(D.dev, &ai, out) != VK_SUCCESS) return false;
  for (uint32_t i = 0; i < n; i++) {
    if (D.set_loader_data) D.set_loader_data(D.dev, out[i]);
    else *(void**)out[i] = *(void**)D.dev;
  }
  return true;
}

bool ensure_buffer(Device& D, VkDeviceSize size) {
  if (D.buf && D.buf_size >= size) return true;
  if (D.buf) {
    D.d.UnmapMemory(D.dev, D.buf_mem);
    D.d.DestroyBuffer(D.dev, D.buf, nullptr);
    D.d.FreeMemory(D.dev, D.buf_mem, nullptr);
    D.buf = VK_NULL_HANDLE;
  }
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (D.d.CreateBuffer(D.dev, &bi, nullptr, &D.buf) != VK_SUCCESS) return false;
  VkMemoryRequirements mr;
  D.d.GetBufferMemoryRequirements(D.dev, D.buf, &mr);
  int32_t t = mem_type(D, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
  if (t < 0) t = mem_type(D, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if (t < 0) return false;
  D.buf_coherent = D.mem.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size; ai.memoryTypeIndex = (uint32_t)t;
  if (D.d.AllocateMemory(D.dev, &ai, nullptr, &D.buf_mem) != VK_SUCCESS) return false;
  D.d.BindBufferMemory(D.dev, D.buf, D.buf_mem, 0);
  D.d.MapMemory(D.dev, D.buf_mem, 0, VK_WHOLE_SIZE, 0, &D.buf_ptr);
  D.buf_size = size;
  return true;
}

QueueState& queue_state(Device& D, VkQueue q) {
  auto it = D.queues.find(q);
  if (it != D.queues.end()) return it->second;
  QueueState s; s.queue = q; s.ordinal = (uint32_t)D.queues.size();
  return D.queues.emplace(q, s).first->second;
}

// ------------------------------------------------------------------ timestamps
void collect_slot(Device& D, QueueState& q, uint32_t k, bool wait) {
  Slot& s = q.ring[k];
  if (!s.pending) return;
  uint64_t v[2] = {0, 0};
  VkQueryResultFlags fl = VK_QUERY_RESULT_64_BIT | (wait ? VK_QUERY_RESULT_WAIT_BIT : 0);
  VkResult r = D.d.GetQueryPoolResults(D.dev, q.qpool, 2 * k, 2, sizeof v, v, sizeof(uint64_t), fl);
  if (r == VK_NOT_READY) return;
  s.pending = false;
  if (r == VK_SUCCESS) D.results.push_back({s.frame, s.batch, q.ordinal, v[0], v[1]});
}

bool ts_ready(Device& D, QueueState& q) {
  if (q.qpool) return true;
  if (q.family >= D.fams.size() || D.fams[q.family].timestampValidBits == 0) return false;
  VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  ci.queryType = VK_QUERY_TYPE_TIMESTAMP; ci.queryCount = 2 * RING;
  if (D.d.CreateQueryPool(D.dev, &ci, nullptr, &q.qpool) != VK_SUCCESS) return false;
  q.ring.resize(RING);
  std::vector<VkCommandBuffer> cbs(2 * RING);
  if (!alloc_cbs(D, q, 2 * RING, cbs.data())) return false;
  for (uint32_t i = 0; i < RING; i++) { q.ring[i].start = cbs[2 * i]; q.ring[i].end = cbs[2 * i + 1]; }
  return true;
}

// returns the slot to use for one batch, with both command buffers recorded
Slot* next_slot(Device& D, QueueState& q) {
  uint32_t k = q.next;
  q.next = (q.next + 1) % RING;
  collect_slot(D, q, k, true);      // the slot's previous batch must be complete before re-recording
  Slot& s = q.ring[k];
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  D.d.ResetCommandBuffer(s.start, 0);
  D.d.BeginCommandBuffer(s.start, &bi);
  D.d.CmdResetQueryPool(s.start, q.qpool, 2 * k, 2);
  D.d.CmdWriteTimestamp(s.start, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, q.qpool, 2 * k);
  D.d.EndCommandBuffer(s.start);
  D.d.ResetCommandBuffer(s.end, 0);
  D.d.BeginCommandBuffer(s.end, &bi);
  D.d.CmdWriteTimestamp(s.end, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q.qpool, 2 * k + 1);
  D.d.EndCommandBuffer(s.end);
  s.pending = true;
  s.frame = G().presents.load();
  s.batch = D.batches++;
  return &s;
}

bool wrappable(const void* pNext) {
  // a device-group submit carries per-command-buffer masks: leave such batches alone
  for (auto* p = (const VkBaseInStructure*)pNext; p; p = p->pNext)
    if (p->sType == VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO) return false;
  return true;
}

void write_timestamps(Device& D) {
  Global& g = G();
  if (g.ts_path.empty() || D.flushed) return;
  D.flushed = true;
  D.d.DeviceWaitIdle(D.dev);
  for (auto& kv : D.queues)
    for (uint32_t k = 0; k < kv.second.ring.size(); k++) collect_slot(D, kv.second, k, true);
  std::sort(D.results.begin(), D.results.end(), [](const Result& a, const Result& b) { return a.batch < b.batch; });
  FILE* f = fopen(g.ts_path.c_str(), "w");
  if (!f) { vlog("cannot write %s", g.ts_path.c_str()); return; }
  uint32_t valid = 64;
  fprintf(f, "{\"format\": \"covenant-probe-timestamps/1\", \"device\": \"");
  for (const char* c = D.props.deviceName; *c; c++) { if (*c == '"' || *c == '\\') fputc('\\', f); fputc(*c, f); }
  fprintf(f, "\", \"timestamp_period_ns\": %.9g, \"presents\": %llu, \"queues\": %zu,\n \"batches\": [",
          (double)D.props.limits.timestampPeriod, (unsigned long long)g.presents.load(), D.queues.size());
  (void)valid;
  for (size_t i = 0; i < D.results.size(); i++) {
    auto& r = D.results[i];
    fprintf(f, "%s\n  [%llu, %u, %llu, %llu]", i ? "," : "", (unsigned long long)r.frame, r.queue,
            (unsigned long long)r.t0, (unsigned long long)r.t1);
  }
  fprintf(f, "\n ],\n \"columns\": [\"frame (presents before the submit)\", \"queue ordinal\", \"start ticks (TOP_OF_PIPE)\", \"end ticks (BOTTOM_OF_PIPE)\"]}\n");
  fclose(f);
  if (!g.quiet) vlog("%zu timestamped batches over %llu presents -> %s", D.results.size(), (unsigned long long)g.presents.load(), g.ts_path.c_str());
}

// ------------------------------------------------------------------ hooks
VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, uint32_t n, const VkSubmitInfo* sub, VkFence fence) {
  Global& g = G();
  Device* D;
  { std::lock_guard<std::mutex> lk(g.mu); D = g.devices[key(queue)]; }
  if (g.ts_path.empty()) return D->d.QueueSubmit(queue, n, sub, fence);
  std::lock_guard<std::mutex> lk(g.mu);
  QueueState& q = queue_state(*D, queue);
  if (!ts_ready(*D, q)) return D->d.QueueSubmit(queue, n, sub, fence);
  std::vector<VkSubmitInfo> s2(sub, sub + n);
  std::vector<std::vector<VkCommandBuffer>> cbs(n);
  std::vector<Slot*> used;
  for (uint32_t i = 0; i < n; i++) {
    if (!sub[i].commandBufferCount || !wrappable(sub[i].pNext)) continue;
    Slot* s = next_slot(*D, q);
    cbs[i].push_back(s->start);
    cbs[i].insert(cbs[i].end(), sub[i].pCommandBuffers, sub[i].pCommandBuffers + sub[i].commandBufferCount);
    cbs[i].push_back(s->end);
    s2[i].commandBufferCount = (uint32_t)cbs[i].size();
    s2[i].pCommandBuffers = cbs[i].data();
    used.push_back(s);
  }
  VkResult r = D->d.QueueSubmit(queue, n, s2.data(), fence);
  if (r != VK_SUCCESS)
    for (Slot* s : used) s->pending = false;     // never executed: its queries will never be written
  return r;
}

VkResult submit2(VkQueue queue, uint32_t n, const VkSubmitInfo2* sub, VkFence fence, bool khr) {
  Global& g = G();
  Device* D;
  { std::lock_guard<std::mutex> lk(g.mu); D = g.devices[key(queue)]; }
  auto next = khr && D->d.QueueSubmit2KHR ? D->d.QueueSubmit2KHR : D->d.QueueSubmit2;
  if (g.ts_path.empty()) return next(queue, n, sub, fence);
  std::lock_guard<std::mutex> lk(g.mu);
  QueueState& q = queue_state(*D, queue);
  if (!ts_ready(*D, q)) return next(queue, n, sub, fence);
  std::vector<VkSubmitInfo2> s2(sub, sub + n);
  std::vector<std::vector<VkCommandBufferSubmitInfo>> cbs(n);
  std::vector<Slot*> used;
  for (uint32_t i = 0; i < n; i++) {
    if (!sub[i].commandBufferInfoCount || !wrappable(sub[i].pNext)) continue;
    Slot* s = next_slot(*D, q);
    VkCommandBufferSubmitInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    a.commandBuffer = s->start;
    cbs[i].push_back(a);
    cbs[i].insert(cbs[i].end(), sub[i].pCommandBufferInfos, sub[i].pCommandBufferInfos + sub[i].commandBufferInfoCount);
    a.commandBuffer = s->end;
    cbs[i].push_back(a);
    s2[i].commandBufferInfoCount = (uint32_t)cbs[i].size();
    s2[i].pCommandBufferInfos = cbs[i].data();
    used.push_back(s);
  }
  VkResult r = next(queue, n, s2.data(), fence);
  if (r != VK_SUCCESS)
    for (Slot* s : used) s->pending = false;
  return r;
}
VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) { return submit2(q, n, s, f, false); }
VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit2KHR(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) { return submit2(q, n, s, f, true); }

VKAPI_ATTR VkResult VKAPI_CALL QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pi) {
  Global& g = G();
  Device* D;
  { std::lock_guard<std::mutex> lk(g.mu); D = g.devices[key(queue)]; }
  uint64_t frame = g.presents.load();
  bool hash = g.hash_file && (g.hash_frames.empty() || g.hash_frames.count(frame));
  if (!hash) {
    VkResult r = D->d.QueuePresentKHR(queue, pi);
    g.presents++;
    if (g.fakeclock_tick) g.fakeclock_tick();
    return r;
  }
  std::unique_lock<std::mutex> lk(g.mu);
  QueueState& q = queue_state(*D, queue);
  bool ok = true;
  if (!q.hash_cb) {
    ok = alloc_cbs(*D, q, 1, &q.hash_cb);
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ok = ok && D->d.CreateFence(D->dev, &fi, nullptr, &q.hash_fence) == VK_SUCCESS &&
         D->d.CreateSemaphore(D->dev, &si, nullptr, &q.hash_sem) == VK_SUCCESS;
  }
  struct Item { const Swapchain* sc; uint32_t image; VkDeviceSize off, size; };
  std::vector<Item> items;
  VkDeviceSize total = 0;
  for (uint32_t i = 0; ok && i < pi->swapchainCount; i++) {
    auto it = D->swapchains.find(pi->pSwapchains[i]);
    if (it == D->swapchains.end() || pi->pImageIndices[i] >= it->second.images.size()) continue;
    VkDeviceSize sz = (VkDeviceSize)it->second.extent.width * it->second.extent.height * bytes_per_pixel(it->second.format);
    items.push_back({&it->second, pi->pImageIndices[i], total, sz});
    total += (sz + 255) & ~(VkDeviceSize)255;
  }
  ok = ok && !items.empty() && ensure_buffer(*D, total);
  if (!ok) {
    lk.unlock();
    VkResult r = D->d.QueuePresentKHR(queue, pi);
    g.presents++;
    if (g.fakeclock_tick) g.fakeclock_tick();
    return r;
  }
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  D->d.ResetCommandBuffer(q.hash_cb, 0);
  D->d.BeginCommandBuffer(q.hash_cb, &bi);
  for (auto& it : items) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = it.sc->images[it.image];
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    D->d.CmdPipelineBarrier(q.hash_cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy c{};
    c.bufferOffset = it.off;
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageExtent = {it.sc->extent.width, it.sc->extent.height, 1};
    D->d.CmdCopyImageToBuffer(q.hash_cb, b.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, D->buf, 1, &c);
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    D->d.CmdPipelineBarrier(q.hash_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }
  VkBufferMemoryBarrier hb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  hb.srcQueueFamilyIndex = hb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  hb.buffer = D->buf; hb.size = VK_WHOLE_SIZE;
  D->d.CmdPipelineBarrier(q.hash_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &hb, 0, nullptr);
  D->d.EndCommandBuffer(q.hash_cb);
  std::vector<VkPipelineStageFlags> stages(pi->waitSemaphoreCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.waitSemaphoreCount = pi->waitSemaphoreCount; si.pWaitSemaphores = pi->pWaitSemaphores; si.pWaitDstStageMask = stages.data();
  si.commandBufferCount = 1; si.pCommandBuffers = &q.hash_cb;
  si.signalSemaphoreCount = 1; si.pSignalSemaphores = &q.hash_sem;
  D->d.ResetFences(D->dev, 1, &q.hash_fence);
  VkResult sr = D->d.QueueSubmit(queue, 1, &si, q.hash_fence);
  VkPresentInfoKHR p2 = *pi;
  p2.waitSemaphoreCount = 1; p2.pWaitSemaphores = &q.hash_sem;
  VkResult r = D->d.QueuePresentKHR(queue, sr == VK_SUCCESS ? &p2 : pi);
  if (sr == VK_SUCCESS) {
    D->d.WaitForFences(D->dev, 1, &q.hash_fence, VK_TRUE, UINT64_MAX);
    if (!D->buf_coherent) {
      VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      mr.memory = D->buf_mem; mr.size = VK_WHOLE_SIZE;
      D->d.InvalidateMappedMemoryRanges(D->dev, 1, &mr);
    }
    for (auto& it : items) {
      uint64_t h = fnv1a((const uint8_t*)D->buf_ptr + it.off, (size_t)it.size);
      fprintf(g.hash_file, "frame %llu swapchain %u image %u %ux%u format %d fnv1a64 %016llx\n", (unsigned long long)frame,
              it.sc->index, it.image, it.sc->extent.width, it.sc->extent.height, (int)it.sc->format, (unsigned long long)h);
      if (!g.dump_dir.empty() && bytes_per_pixel(it.sc->format) == 4) {
        char path[1024];
        snprintf(path, sizeof path, "%s/frame%llu_%u.ppm", g.dump_dir.c_str(), (unsigned long long)frame, it.sc->index);
        if (FILE* f = fopen(path, "wb")) {
          uint32_t w = it.sc->extent.width, hgt = it.sc->extent.height;
          bool bgr = it.sc->format == VK_FORMAT_B8G8R8A8_UNORM || it.sc->format == VK_FORMAT_B8G8R8A8_SRGB;
          fprintf(f, "P6\n%u %u\n255\n", w, hgt);
          const uint8_t* px = (const uint8_t*)D->buf_ptr + it.off;
          for (size_t k = 0; k < (size_t)w * hgt; k++) {
            uint8_t rgb[3] = {px[4 * k + (bgr ? 2 : 0)], px[4 * k + 1], px[4 * k + (bgr ? 0 : 2)]};
            fwrite(rgb, 1, 3, f);
          }
          fclose(f);
        }
      }
    }
    fflush(g.hash_file);
    g.hashed++;
  }
  g.presents++;
  if (g.fakeclock_tick) g.fakeclock_tick();
  return r;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSwapchainKHR(VkDevice dev, const VkSwapchainCreateInfoKHR* ci, const VkAllocationCallbacks* a,
                                                  VkSwapchainKHR* out) {
  Global& g = G();
  Device* D;
  { std::lock_guard<std::mutex> lk(g.mu); D = g.devices[key(dev)]; }
  VkSwapchainCreateInfoKHR c2 = *ci;
  if (g.hash_file) c2.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  VkResult r = D->d.CreateSwapchainKHR(dev, &c2, a, out);
  if (r != VK_SUCCESS) return r;
  Swapchain s;
  s.format = ci->imageFormat; s.extent = ci->imageExtent;
  uint32_t n = 0;
  D->d.GetSwapchainImagesKHR(dev, *out, &n, nullptr);
  s.images.resize(n);
  D->d.GetSwapchainImagesKHR(dev, *out, &n, s.images.data());
  std::lock_guard<std::mutex> lk(g.mu);
  s.index = D->swapchain_count++;
  D->swapchains[*out] = s;
  return r;
}

VKAPI_ATTR void VKAPI_CALL DestroySwapchainKHR(VkDevice dev, VkSwapchainKHR sc, const VkAllocationCallbacks* a) {
  Global& g = G();
  Device* D;
  {
    std::lock_guard<std::mutex> lk(g.mu);
    D = g.devices[key(dev)];
    D->swapchains.erase(sc);
  }
  D->d.DestroySwapchainKHR(dev, sc, a);
}

void record_queue(Device* D, VkQueue q, uint32_t family, uint32_t index) {
  std::lock_guard<std::mutex> lk(G().mu);
  QueueState& s = queue_state(*D, q);
  s.family = family; s.index = index;
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice dev, uint32_t fam, uint32_t idx, VkQueue* q) {
  Device* D;
  { std::lock_guard<std::mutex> lk(G().mu); D = G().devices[key(dev)]; }
  D->d.GetDeviceQueue(dev, fam, idx, q);
  if (*q) record_queue(D, *q, fam, idx);
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue2(VkDevice dev, const VkDeviceQueueInfo2* qi, VkQueue* q) {
  Device* D;
  { std::lock_guard<std::mutex> lk(G().mu); D = G().devices[key(dev)]; }
  D->d.GetDeviceQueue2(dev, qi, q);
  if (*q) record_queue(D, *q, qi->queueFamilyIndex, qi->queueIndex);
}

void release(Device* D) {
  write_timestamps(*D);
  D->d.DeviceWaitIdle(D->dev);
  for (auto& kv : D->queues) {
    auto& q = kv.second;
    if (q.hash_fence) D->d.DestroyFence(D->dev, q.hash_fence, nullptr);
    if (q.hash_sem) D->d.DestroySemaphore(D->dev, q.hash_sem, nullptr);
    if (q.qpool) D->d.DestroyQueryPool(D->dev, q.qpool, nullptr);
    if (q.pool) D->d.DestroyCommandPool(D->dev, q.pool, nullptr);
  }
  if (D->buf) {
    D->d.UnmapMemory(D->dev, D->buf_mem);
    D->d.DestroyBuffer(D->dev, D->buf, nullptr);
    D->d.FreeMemory(D->dev, D->buf_mem, nullptr);
  }
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice dev, const VkAllocationCallbacks* a) {
  Global& g = G();
  Device* D;
  {
    std::lock_guard<std::mutex> lk(g.mu);
    auto it = g.devices.find(key(dev));
    D = it->second;
    g.devices.erase(it);
  }
  release(D);
  auto destroy = D->d.DestroyDevice;
  delete D;
  destroy(dev, a);
}

void at_exit() {
  Global& g = G();
  std::vector<Device*> ds;
  {
    std::lock_guard<std::mutex> lk(g.mu);
    for (auto& kv : g.devices) ds.push_back(kv.second);
  }
  for (auto* D : ds) write_timestamps(*D);       // programs that never destroy their device
  if (g.hash_file) { fclose(g.hash_file); g.hash_file = nullptr; }
  if (!g.quiet) vlog("presents %llu, hashed %llu", (unsigned long long)g.presents.load(), (unsigned long long)g.hashed);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo* ci, const VkAllocationCallbacks* a, VkDevice* out) {
  VkLayerDeviceCreateInfo* link = nullptr;
  PFN_vkSetDeviceLoaderData sld = nullptr;
  for (auto* p = (VkLayerDeviceCreateInfo*)ci->pNext; p; p = (VkLayerDeviceCreateInfo*)p->pNext) {
    if (p->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO) continue;
    if (p->function == VK_LAYER_LINK_INFO && !link) link = p;
    else if (p->function == VK_LOADER_DATA_CALLBACK) sld = p->u.pfnSetDeviceLoaderData;
  }
  if (!link) return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkGetInstanceProcAddr gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
  link->u.pLayerInfo = link->u.pLayerInfo->pNext;
  auto create = (PFN_vkCreateDevice)gipa(VK_NULL_HANDLE, "vkCreateDevice");
  VkResult r = create(phys, ci, a, out);
  if (r != VK_SUCCESS) return r;
  Device* D = new Device();
  D->dev = *out; D->phys = phys; D->gdpa = gdpa; D->set_loader_data = sld;
#define LOAD(n) D->d.n = (PFN_vk##n)gdpa(*out, "vk" #n);
  DEV_FNS(LOAD)
#undef LOAD
  Global& g = G();
  std::lock_guard<std::mutex> lk(g.mu);
  auto ii = g.phys_inst.find(phys);
  if (ii != g.phys_inst.end()) {
    Instance& I = g.instances[ii->second];
    if (I.mem_props) I.mem_props(phys, &D->mem);
    if (I.props) I.props(phys, &D->props);
    if (I.qfam) {
      uint32_t n = 0;
      I.qfam(phys, &n, nullptr);
      D->fams.resize(n);
      I.qfam(phys, &n, D->fams.data());
    }
  }
  g.devices[key(*out)] = D;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(VkInstance inst, uint32_t* n, VkPhysicalDevice* p);

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo* ci, const VkAllocationCallbacks* a, VkInstance* out) {
  load_env();
  auto* info = (VkLayerInstanceCreateInfo*)ci->pNext;
  while (info && !(info->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && info->function == VK_LAYER_LINK_INFO))
    info = (VkLayerInstanceCreateInfo*)info->pNext;
  if (!info) return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkGetInstanceProcAddr gipa = info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  info->u.pLayerInfo = info->u.pLayerInfo->pNext;
  auto create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
  VkResult r = create(ci, a, out);
  if (r != VK_SUCCESS) return r;
  Instance I;
  I.gipa = gipa;
  I.mem_props = (PFN_vkGetPhysicalDeviceMemoryProperties)gipa(*out, "vkGetPhysicalDeviceMemoryProperties");
  I.props = (PFN_vkGetPhysicalDeviceProperties)gipa(*out, "vkGetPhysicalDeviceProperties");
  I.qfam = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa(*out, "vkGetPhysicalDeviceQueueFamilyProperties");
  static bool registered = false;
  std::lock_guard<std::mutex> lk(G().mu);
  G().instances[key(*out)] = I;
  if (!registered) { registered = true; atexit(at_exit); }
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDevices(VkInstance inst, uint32_t* n, VkPhysicalDevice* p) {
  PFN_vkGetInstanceProcAddr gipa;
  { std::lock_guard<std::mutex> lk(G().mu); gipa = G().instances[key(inst)].gipa; }
  auto next = (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
  VkResult r = next(inst, n, p);
  if (p && (r == VK_SUCCESS || r == VK_INCOMPLETE)) {
    std::lock_guard<std::mutex> lk(G().mu);
    for (uint32_t i = 0; i < *n; i++) G().phys_inst[p[i]] = key(inst);
  }
  return r;
}

VKAPI_ATTR VkResult VKAPI_CALL EnumeratePhysicalDeviceGroups(VkInstance inst, uint32_t* n, VkPhysicalDeviceGroupProperties* p) {
  PFN_vkGetInstanceProcAddr gipa;
  { std::lock_guard<std::mutex> lk(G().mu); gipa = G().instances[key(inst)].gipa; }
  auto next = (PFN_vkEnumeratePhysicalDeviceGroups)gipa(inst, "vkEnumeratePhysicalDeviceGroups");
  if (!next) next = (PFN_vkEnumeratePhysicalDeviceGroups)gipa(inst, "vkEnumeratePhysicalDeviceGroupsKHR");
  VkResult r = next(inst, n, p);
  if (p && (r == VK_SUCCESS || r == VK_INCOMPLETE)) {
    std::lock_guard<std::mutex> lk(G().mu);
    for (uint32_t i = 0; i < *n; i++)
      for (uint32_t j = 0; j < p[i].physicalDeviceCount; j++) G().phys_inst[p[i].physicalDevices[j]] = key(inst);
  }
  return r;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance inst, const VkAllocationCallbacks* a) {
  PFN_vkGetInstanceProcAddr gipa;
  {
    std::lock_guard<std::mutex> lk(G().mu);
    gipa = G().instances[key(inst)].gipa;
    G().instances.erase(key(inst));
  }
  auto destroy = (PFN_vkDestroyInstance)gipa(inst, "vkDestroyInstance");
  destroy(inst, a);
}

PFN_vkVoidFunction device_hook(const char* n) {
#define H(x) if (!strcmp(n, "vk" #x)) return (PFN_vkVoidFunction)x;
  H(QueueSubmit) H(QueueSubmit2) H(QueueSubmit2KHR) H(QueuePresentKHR) H(CreateSwapchainKHR) H(DestroySwapchainKHR)
  H(GetDeviceQueue) H(GetDeviceQueue2) H(DestroyDevice)
#undef H
  return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice dev, const char* n);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance inst, const char* n) {
  if (!strcmp(n, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)GetInstanceProcAddr;
  if (!strcmp(n, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)GetDeviceProcAddr;
  if (!strcmp(n, "vkCreateInstance")) return (PFN_vkVoidFunction)CreateInstance;
  if (!strcmp(n, "vkDestroyInstance")) return (PFN_vkVoidFunction)DestroyInstance;
  if (!strcmp(n, "vkCreateDevice")) return (PFN_vkVoidFunction)CreateDevice;
  if (!strcmp(n, "vkEnumeratePhysicalDevices")) return (PFN_vkVoidFunction)EnumeratePhysicalDevices;
  if (!strcmp(n, "vkEnumeratePhysicalDeviceGroups") || !strcmp(n, "vkEnumeratePhysicalDeviceGroupsKHR"))
    return (PFN_vkVoidFunction)EnumeratePhysicalDeviceGroups;
  if (auto f = device_hook(n)) return f;
  if (!inst) return nullptr;
  PFN_vkGetInstanceProcAddr gipa;
  {
    std::lock_guard<std::mutex> lk(G().mu);
    auto it = G().instances.find(key(inst));
    if (it == G().instances.end()) return nullptr;
    gipa = it->second.gipa;
  }
  return gipa(inst, n);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice dev, const char* n) {
  if (!strcmp(n, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)GetDeviceProcAddr;
  PFN_vkGetDeviceProcAddr gdpa;
  Device* D;
  {
    std::lock_guard<std::mutex> lk(G().mu);
    auto it = G().devices.find(key(dev));
    if (it == G().devices.end()) return nullptr;
    D = it->second;
    gdpa = D->gdpa;
  }
  auto f = device_hook(n);
  if (f) {
    // only claim functions the next layer/driver offers (e.g. vkQueueSubmit2 on a 1.2 device)
    return gdpa(dev, n) ? f : nullptr;
  }
  return gdpa(dev, n);
}

}  // namespace

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance inst, const char* n) {
  return GetInstanceProcAddr(inst, n);
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice dev, const char* n) {
  return GetDeviceProcAddr(dev, n);
}

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v) {
  if (v->loaderLayerInterfaceVersion > 2) v->loaderLayerInterfaceVersion = 2;
  v->pfnGetInstanceProcAddr = GetInstanceProcAddr;
  v->pfnGetDeviceProcAddr = GetDeviceProcAddr;
  v->pfnGetPhysicalDeviceProcAddr = nullptr;
  return VK_SUCCESS;
}
