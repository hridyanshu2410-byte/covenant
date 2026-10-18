// COVENANT model driver: a stand-in libvulkan that needs no GPU.
//
// Every Vulkan call is recorded as one JSON line in $COVENANT_TRACE (default
// covenant-trace.jsonl) and answered with fake, unique handles and plausible
// properties. Host writes to mapped memory are observed with page protection.
// See README.md for the trace format.
#include <vulkan/vulkan.h>

#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <dirent.h>

#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct FlagName { uint64_t bit; const char* name; };
struct ProcEntry { const char* name; PFN_vkVoidFunction fn; };

// ---------------------------------------------------------------- utilities

namespace {

std::recursive_mutex g_mu;
FILE* g_out = nullptr;
uint64_t g_seq = 0;
size_t g_page = 4096;

// --- SHA-1 (for shader identity) ---
struct Sha1 {
  uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
  uint8_t buf[64]; size_t blen = 0; uint64_t total = 0;
  static uint32_t rol(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
  void block(const uint8_t* p) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) w[i] = (p[4*i] << 24) | (p[4*i+1] << 16) | (p[4*i+2] << 8) | p[4*i+3];
    for (int i = 16; i < 80; i++) w[i] = rol(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; i++) {
      uint32_t f, k;
      if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
      else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
      else { f = b ^ c ^ d; k = 0xCA62C1D6; }
      uint32_t t = rol(a, 5) + f + e + k + w[i];
      e = d; d = c; c = rol(b, 30); b = a; a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
  }
  void update(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data; total += n;
    while (n--) { buf[blen++] = *p++; if (blen == 64) { block(buf); blen = 0; } }
  }
  std::string hex() {
    uint64_t bits = total * 8; uint8_t pad = 0x80; update(&pad, 1);
    uint8_t z = 0; while (blen != 56) update(&z, 1);
    for (int i = 7; i >= 0; i--) { uint8_t c = (uint8_t)(bits >> (8 * i)); update(&c, 1); }
    char out[41];
    for (int i = 0; i < 5; i++) snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 40);
  }
};

// --- handles ---
std::unordered_map<uint64_t, std::string> g_ids;
std::map<std::string, int> g_count;
uint64_t g_next = 0x10000;

uint64_t mk(const char* kind) {
  uint64_t h = (g_next += 0x10);
  g_ids[h] = std::string(kind) + "#" + std::to_string(++g_count[kind]);
  return h;
}
#define H(x) ((uint64_t)(x))
template <class T> T as(uint64_t h) { return (T)(h); }

// --- JSON writer ---
struct W {
  std::string s; std::vector<bool> first; bool ak = false;
  void sep() {
    if (ak) { ak = false; return; }
    if (!first.empty()) { if (!first.back()) s += ','; first.back() = false; }
  }
  W& ob() { sep(); s += '{'; first.push_back(true); return *this; }
  W& oe() { s += '}'; first.pop_back(); return *this; }
  W& ab() { sep(); s += '['; first.push_back(true); return *this; }
  W& ae() { s += ']'; first.pop_back(); return *this; }
  W& k(const char* key) { sep(); s += '"'; s += key; s += "\":"; ak = true; return *this; }
  W& str(const char* v) {
    if (!v) return nul();
    sep(); s += '"';
    for (const char* p = v; *p; p++) {
      unsigned char c = *p;
      if (c == '"' || c == '\\') { s += '\\'; s += c; }
      else if (c < 0x20) { char b[8]; snprintf(b, 8, "\\u%04x", c); s += b; }
      else s += c;
    }
    s += '"'; return *this;
  }
  W& str(const std::string& v) { return str(v.c_str()); }
  W& u(uint64_t v) { sep(); s += std::to_string(v); return *this; }
  W& i(int64_t v) { sep(); s += std::to_string(v); return *this; }
  W& d(double v) {   // JSON has no NaN/Inf (union readings of VkClearValue often are): write null
    if (v != v || v > 1e308 || v < -1e308) return nul();
    sep(); char b[32]; snprintf(b, 32, "%g", v); s += b; return *this;
  }
  W& b(bool v) { sep(); s += v ? "true" : "false"; return *this; }
  W& nul() { sep(); s += "null"; return *this; }
  W& h(uint64_t v) {
    if (!v) return nul();
    auto it = g_ids.find(v);
    if (it != g_ids.end()) return str(it->second);
    char b[40]; snprintf(b, 40, "unknown#%llx", (unsigned long long)v); return str(b);
  }
  W& en(const char* name, int64_t v) { return name ? str(name) : i(v); }
  W& fl(const FlagName* t, uint64_t v) {
    ab();
    for (; t->name; t++) if (v & t->bit) { str(t->name); v &= ~t->bit; }
    if (v) u(v);
    return ae();
  }
  W& family(uint32_t f) { return f == VK_QUEUE_FAMILY_IGNORED ? nul() : u(f); }
  W& hex(const void* data, size_t n) {
    std::string x; char b[3];
    for (size_t i = 0; i < n; i++) { snprintf(b, 3, "%02x", ((const uint8_t*)data)[i]); x += b; }
    return str(x);
  }
};

}  // namespace

// generated tables and stubs need these two before inclusion (see bottom)
extern "C" void covvk_unimplemented(const char* name);
extern "C" void covvk_record_cmd(const char* name, uint64_t cb);
extern "C" void covvk_unimplemented_cmd(const char* name, uint64_t cb);
template <class T> T covvk_fake_handle(const char* kind);

namespace {
#include "covvk_enums.inc"
}

namespace {

// --- host-write tracking via page protection ---
struct Region {
  std::atomic<char*> base{nullptr};
  size_t len = 0;
  std::atomic<int> dirty{0};
  volatile uint8_t* bits = nullptr;
  uint8_t* shadow = nullptr;     // bytes as of the last flush: host_write records carry the diff
  uint64_t mem = 0;
  // the threads whose store faulted a page of this region in the
  // current interval (the first store per page only: a later store to a page that is
  // already writable does not fault), at most 4, then tid_over
  std::atomic<int> ntid{0};
  std::atomic<long> tids[4];
  std::atomic<int> tid_over{0};
};
const size_t MAX_WRITE_BYTES = 256 * 1024;   // bytes of content per host_write record
const size_t MAX_WRITE_RUNS = 1 << 20;       // byte runs per record (content or length-only)
const int MAXREG = 4096;
Region g_reg[MAXREG];
std::atomic<int> g_nreg{0};
struct sigaction g_old_segv;
bool g_handler = false;
bool g_track = true;
long g_last_call_tid = 0;       // thread of the previous Vulkan call (under g_mu)
bool g_handler_warned = false;

void segv(int sig, siginfo_t* si, void* uc) {
  char* a = (char*)si->si_addr;
  int n = g_nreg.load();
  for (int i = 0; i < n; i++) {
    char* b = g_reg[i].base.load();
    if (b && a >= b && a < b + g_reg[i].len) {
      size_t pg = (size_t)(a - b) / g_page;
      volatile uint8_t* bits = g_reg[i].bits;
      if (bits) bits[pg] = 1;
      g_reg[i].dirty.store(1);
      long t = (long)syscall(SYS_gettid);          // async-signal-safe
      Region& r = g_reg[i];
      bool seen = false;
      int k = r.ntid.load();
      for (int j = 0; j < k && j < 4; j++) if (r.tids[j].load() == t) { seen = true; break; }
      if (!seen) { int j = r.ntid.fetch_add(1); if (j < 4) r.tids[j].store(t); else r.tid_over.store(1); }
      mprotect(b + pg * g_page, g_page, PROT_READ | PROT_WRITE);
      return;
    }
  }
  if (g_old_segv.sa_flags & SA_SIGINFO) {
    if (g_old_segv.sa_sigaction) { g_old_segv.sa_sigaction(sig, si, uc); return; }
  } else if (g_old_segv.sa_handler != SIG_DFL && g_old_segv.sa_handler != SIG_IGN) {
    g_old_segv.sa_handler(sig); return;
  }
  signal(sig, SIG_DFL);   // returning re-raises the fault with the default action
}

void open_trace();
W rec(const char* fn);
void emit(W& w);

int live_threads() {
  // threads of the process at this flush (/proc/self/task entries)
  DIR* d = opendir("/proc/self/task");
  if (!d) return -1;
  int n = 0;
  while (struct dirent* e = readdir(d)) if (e->d_name[0] != '.') n++;
  closedir(d);
  return n;
}

void flush_host_writes() {
  int n = g_nreg.load();
  long cur_tid = (long)syscall(SYS_gettid);
  if (n > 0 && g_handler && !g_handler_warned) {
    // the capture needs our SIGSEGV handler; an application that installs its own
    // (crash reporters) takes the faults from here on, and the capture is incomplete
    struct sigaction cur = {};
    if (sigaction(SIGSEGV, nullptr, &cur) == 0 && !((cur.sa_flags & SA_SIGINFO) && cur.sa_sigaction == segv)) {
      g_handler_warned = true;
      W w = rec("covenant_capture_warning");
      w.k("reason").str("SIGSEGV handler replaced by the application");
      emit(w);
    }
  }
  int threads = -2;
  for (int i = 0; i < n; i++) {
    Region& r = g_reg[i];
    if (!r.base.load() || !r.dirty.exchange(0)) continue;
    size_t pages = r.len / g_page;
    std::vector<uint64_t> offs;
    for (size_t p = 0; p < pages; p++) if (r.bits[p]) { offs.push_back(p * g_page); r.bits[p] = 0; }
    if (offs.empty()) continue;
    // re-protect before diffing. A write that lands while this flush runs is
    // either before the re-protect (then it is in this diff) or faults and marks its page
    // for the next record. Re-protecting after the diff would leave written pages writable
    // during the diff: a concurrent write there would set no bit and surface in a later
    // record, attributed to a later interval than the one it happened in.
    mprotect(r.base.load(), r.len, PROT_READ);
    W w = rec("host_write");
    w.k("memory").h(r.mem).k("offset").u(offs[0]).k("page_size").u(g_page);
    w.k("offsets").ab(); for (auto o : offs) w.u(o); w.ae();
    {   // who wrote, and the threads of the calls that bound the interval
      int k = r.ntid.exchange(0);
      w.k("fault_tids").ab();
      for (int j = 0; j < k && j < 4; j++) w.i((int)r.tids[j].load());
      w.ae();
      if (r.tid_over.exchange(0)) w.k("fault_tids_overflow").b(true);
      w.k("call_tids").ab().i((int)g_last_call_tid).i((int)cur_tid).ae();
      if (threads == -2) threads = live_threads();
      w.k("threads").i(threads);
    }
    if (r.shadow) {
      // byte runs that differ from the shadow (gaps < 16 bytes merged): [offset, hex] while
      // the content budget lasts, then [offset, "", length] without content, so the
      // checker keeps exact spans; only past MAX_WRITE_RUNS runs is the record truncated
      const uint8_t* cur = (const uint8_t*)r.base.load();
      size_t total = 0, runs = 0; bool trunc = false, nocontent = false; size_t trunc_at = 0;
      w.k("writes").ab();
      for (auto o : offs) {
        size_t i = o, end = o + g_page;
        while (i < end) {
          if (cur[i] == r.shadow[i]) { i++; continue; }
          size_t j = i + 1, last = i;
          while (j < end && j - last < 16) { if (cur[j] != r.shadow[j]) last = j; j++; }
          size_t n = last + 1 - i;
          if (runs >= MAX_WRITE_RUNS) { trunc = true; trunc_at = i; break; }
          if (!nocontent && total + n > MAX_WRITE_BYTES) nocontent = true;
          if (nocontent) w.ab().u(i).str("").u(n).ae();
          else { w.ab().u(i).hex(cur + i, n).ae(); total += n; }
          memcpy(r.shadow + i, cur + i, n);
          runs++; i = last + 1;
        }
        if (trunc) break;
      }
      w.ae();
      if (nocontent) w.k("content_truncated").b(true);
      if (trunc) {
        w.k("writes_truncated").b(true).k("truncated_at").u(trunc_at);
        memcpy(r.shadow, cur, r.len);
      }
    }
    emit(w);
  }
}

void pre_call() {
  open_trace();
  flush_host_writes();
  g_last_call_tid = (long)syscall(SYS_gettid);
}

#define ENTER std::lock_guard<std::recursive_mutex> _covvk_lk(g_mu); pre_call();

void finish() {
  std::lock_guard<std::recursive_mutex> lk(g_mu);
  if (!g_out) return;
  flush_host_writes();
  W w = rec("covenant_trace_end"); emit(w);
  fclose(g_out); g_out = nullptr;
}

void open_trace() {
  if (g_out) return;
  // One driver per process. An app that loads libvulkan.so and libvulkan.so.1 as two
  // different files (copies, e.g. volk) gets two independent drivers appending to one
  // trace with conflicting handle numbers; refuse the second one.
  char me[64];
  snprintf(me, sizeof me, "%d:%p", (int)getpid(), (void*)&g_out);
  const char* other = getenv("COVVK_DRIVER_ACTIVE");
  if (other && atoi(other) == (int)getpid() && strcmp(other, me) != 0) {
    fprintf(stderr, "covvk: a second copy of the COVENANT driver was loaded into this process "
                    "(libvulkan.so and libvulkan.so.1 are different files?); make them hard links "
                    "to libcovvk.so. Aborting so the trace is not corrupted.\n");
    abort();
  }
  setenv("COVVK_DRIVER_ACTIVE", me, 1);
  const char* p = getenv("COVENANT_TRACE");
  g_out = fopen(p && *p ? p : "covenant-trace.jsonl", "w");
  if (!g_out) { perror("covvk: cannot open trace"); g_out = stderr; }
  g_page = (size_t)sysconf(_SC_PAGESIZE);
  const char* t = getenv("COVENANT_NO_HOST_TRACKING");
  g_track = !(t && *t && *t != '0');
  W w = rec("covenant_trace_begin");
  w.k("version").u(1).k("pid").i(getpid()).k("header_version").u(VK_HEADER_VERSION)
   .k("host_write_tracking").b(g_track).k("page_size").u(g_page);
  emit(w);
  atexit(finish);
}

W rec(const char* fn) {
  W w; w.ob(); w.k("seq").u(++g_seq); w.k("fn").str(fn); return w;
}

void emit(W& w) {
  w.oe();
  fputs(w.s.c_str(), g_out); fputc('\n', g_out); fflush(g_out);
}

// stage/access names: legacy flag bits share their values with the sync2 bits
W& stages(W& w, uint64_t v) { return w.fl(flags_VkPipelineStageFlagBits2, v); }
W& access(W& w, uint64_t v) { return w.fl(flags_VkAccessFlagBits2, v); }

template <class T> const T* find_chain(const void* pNext, VkStructureType st) {
  for (auto* p = (const VkBaseInStructure*)pNext; p; p = p->pNext)
    if (p->sType == st) return (const T*)p;
  return nullptr;
}

// --- driver state ---
struct MemObj { VkDeviceSize size = 0; uint32_t type = 0; char* base = nullptr; size_t len = 0; int region = -1; };
std::unordered_map<uint64_t, MemObj> g_mem;
struct Swap { std::vector<uint64_t> images; uint32_t next = 0; };
std::unordered_map<uint64_t, Swap> g_swap;
std::unordered_map<uint64_t, VkDeviceSize> g_bufsize;
std::unordered_map<uint64_t, std::pair<uint64_t, VkDeviceSize>> g_bufbind;   // buffer -> (memory, offset)
// image -> buffer copies recorded per command buffer: filled with a probe
// ramp at submit (the model has no GPU; see readback_fill)
struct ReadOp { uint64_t image, buffer; VkBufferImageCopy region; };
std::unordered_map<uint64_t, std::vector<ReadOp>> g_readops;
std::unordered_map<uint64_t, VkImageCreateInfo> g_image;
std::unordered_map<uint64_t, std::string> g_module_sha;
std::map<std::pair<uint32_t, uint32_t>, uint64_t> g_queues;
std::set<std::string> g_warned;
uint64_t g_instance = 0, g_phys = 0, g_device = 0;
uint64_t g_bda = 0x100000000ULL;

// ---- device addresses (stable per buffer) and acceleration structures
std::unordered_map<uint64_t, VkDeviceAddress> g_bufaddr;   // buffer -> address
std::map<VkDeviceAddress, uint64_t> g_addrbuf;             // address -> buffer
struct AccelStruct { uint64_t buffer = 0; VkDeviceSize offset = 0, size = 0; VkAccelerationStructureTypeKHR type{}; VkDeviceAddress addr = 0; };
std::unordered_map<uint64_t, AccelStruct> g_as;
std::unordered_map<VkDeviceAddress, uint64_t> g_asaddr;   // AS device address -> AS
// TLAS builds whose instance references are read from host-visible memory at submit
struct TlasPending { uint64_t seq; uint32_t build; uint64_t buffer; VkDeviceSize offset; uint32_t count; bool pointers; };
std::unordered_map<uint64_t, std::vector<TlasPending>> g_tlas_pending;
std::unordered_map<uint64_t, uint32_t> g_query_type;      // query pool -> VkQueryType

VkDeviceAddress buf_addr(uint64_t b) {
  auto it = g_bufaddr.find(b);
  if (it != g_bufaddr.end()) return it->second;
  VkDeviceAddress a = g_bda; g_bda += ((g_bufsize[b] + 0xFFFF) & ~0xFFFFull) + 0x10000;
  g_bufaddr[b] = a; g_addrbuf[a] = b;
  return a;
}
// address -> (buffer, offset); buffer 0 when no buffer holds it
std::pair<uint64_t, VkDeviceSize> resolve_addr(VkDeviceAddress a) {
  auto it = g_addrbuf.upper_bound(a);
  if (it == g_addrbuf.begin()) return {0, 0};
  --it;
  VkDeviceSize sz = std::max<VkDeviceSize>(g_bufsize[it->second], 1);
  if (a - it->first < sz) return {it->second, a - it->first};
  return {0, 0};
}
// {"buffer", "offset", "size"} of a device-address range, or {"address", "unresolved": true}
void w_devrange(W& w, const char* key, VkDeviceAddress a, VkDeviceSize size) {
  w.k(key);
  if (!a) { w.nul(); return; }
  auto r = resolve_addr(a);
  w.ob();
  if (r.first) w.k("buffer").h(r.first).k("offset").u(r.second);
  else w.k("address").u(a).k("unresolved").b(true);
  w.k("size").u(size).oe();
}
// sizes the fake device reports (and assumes at build time): a function of the primitive count
VkDeviceSize as_size(uint64_t prims) { return (4096 + 256 * prims + 255) & ~VkDeviceSize(255); }
VkDeviceSize as_scratch(uint64_t prims, bool update) { return ((update ? 512 : 1024) + 128 * prims + 255) & ~VkDeviceSize(255); }

std::pair<uint32_t, uint32_t> surface_extent() {
  uint32_t w = 800, h = 600;
  if (const char* e = getenv("COVENANT_SURFACE_EXTENT")) sscanf(e, "%ux%u", &w, &h);
  return {w, h};
}

const VkMemoryPropertyFlags kMemTypes[] = {
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
};
const uint32_t kMemHeap[] = {0, 1, 1, 0};

uint32_t format_size(VkFormat f) {
  switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SRGB: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_D16_UNORM: return 2;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return 8;
    default: return 4;
  }
}

VkDeviceSize image_bytes(const VkImageCreateInfo& ci) {
  VkDeviceSize t = 0;
  uint32_t w = ci.extent.width, h = ci.extent.height, d = ci.extent.depth;
  for (uint32_t m = 0; m < std::max(1u, ci.mipLevels); m++) {
    t += (VkDeviceSize)std::max(1u, w >> m) * std::max(1u, h >> m) * std::max(1u, d >> m);
  }
  t *= format_size(ci.format) * std::max(1u, ci.arrayLayers) * std::max(1, (int)ci.samples);
  return (t + 4095) & ~VkDeviceSize(4095);
}

void fill_bools(void* s, size_t size) {
  // feature structs: sType, pNext, then only VkBool32 members
  size_t off = offsetof(VkPhysicalDeviceVulkan11Features, storageBuffer16BitAccess);
  for (size_t o = off; o + 4 <= size; o += 4) *(VkBool32*)((char*)s + o) = VK_TRUE;
}

void fill_limits(VkPhysicalDeviceLimits& l) {
  memset(&l, 0, sizeof l);
  l.maxImageDimension1D = l.maxImageDimension2D = l.maxImageDimensionCube = 16384;
  l.maxImageDimension3D = 2048; l.maxImageArrayLayers = 2048;
  l.maxTexelBufferElements = 1u << 27; l.maxUniformBufferRange = 65536;
  l.maxStorageBufferRange = 1u << 30; l.maxPushConstantsSize = 256;
  l.maxMemoryAllocationCount = 4096; l.maxSamplerAllocationCount = 4000;
  l.bufferImageGranularity = 1; l.sparseAddressSpaceSize = 1ull << 40;
  l.maxBoundDescriptorSets = 8;
  l.maxPerStageDescriptorSamplers = l.maxPerStageDescriptorUniformBuffers =
  l.maxPerStageDescriptorStorageBuffers = l.maxPerStageDescriptorSampledImages =
  l.maxPerStageDescriptorStorageImages = l.maxPerStageDescriptorInputAttachments = 1u << 20;
  l.maxPerStageResources = 1u << 20;
  l.maxDescriptorSetSamplers = l.maxDescriptorSetUniformBuffers = l.maxDescriptorSetStorageBuffers =
  l.maxDescriptorSetSampledImages = l.maxDescriptorSetStorageImages = l.maxDescriptorSetInputAttachments = 1u << 20;
  l.maxDescriptorSetUniformBuffersDynamic = 15; l.maxDescriptorSetStorageBuffersDynamic = 16;
  l.maxVertexInputAttributes = 32; l.maxVertexInputBindings = 32;
  l.maxVertexInputAttributeOffset = 2047; l.maxVertexInputBindingStride = 2048;
  l.maxVertexOutputComponents = 128;
  l.maxTessellationGenerationLevel = 64; l.maxTessellationPatchSize = 32;
  l.maxTessellationControlPerVertexInputComponents = l.maxTessellationControlPerVertexOutputComponents = 128;
  l.maxTessellationControlPerPatchOutputComponents = 120; l.maxTessellationControlTotalOutputComponents = 4096;
  l.maxTessellationEvaluationInputComponents = l.maxTessellationEvaluationOutputComponents = 128;
  l.maxGeometryShaderInvocations = 32; l.maxGeometryInputComponents = 128;
  l.maxGeometryOutputComponents = 128; l.maxGeometryOutputVertices = 256;
  l.maxGeometryTotalOutputComponents = 1024;
  l.maxFragmentInputComponents = 128; l.maxFragmentOutputAttachments = 8;
  l.maxFragmentDualSrcAttachments = 1; l.maxFragmentCombinedOutputResources = 1u << 20;
  l.maxComputeSharedMemorySize = 49152;
  l.maxComputeWorkGroupCount[0] = l.maxComputeWorkGroupCount[1] = l.maxComputeWorkGroupCount[2] = 65535;
  l.maxComputeWorkGroupInvocations = 1024;
  l.maxComputeWorkGroupSize[0] = 1024; l.maxComputeWorkGroupSize[1] = 1024; l.maxComputeWorkGroupSize[2] = 64;
  l.subPixelPrecisionBits = 8; l.subTexelPrecisionBits = 8; l.mipmapPrecisionBits = 8;
  l.maxDrawIndexedIndexValue = UINT32_MAX; l.maxDrawIndirectCount = UINT32_MAX;
  l.maxSamplerLodBias = 16; l.maxSamplerAnisotropy = 16;
  l.maxViewports = 16; l.maxViewportDimensions[0] = l.maxViewportDimensions[1] = 16384;
  l.viewportBoundsRange[0] = -32768; l.viewportBoundsRange[1] = 32767; l.viewportSubPixelBits = 8;
  l.minMemoryMapAlignment = 64; l.minTexelBufferOffsetAlignment = 16;
  l.minUniformBufferOffsetAlignment = 64; l.minStorageBufferOffsetAlignment = 16;
  l.minTexelOffset = -8; l.maxTexelOffset = 7; l.minTexelGatherOffset = -32; l.maxTexelGatherOffset = 31;
  l.minInterpolationOffset = -0.5f; l.maxInterpolationOffset = 0.4375f; l.subPixelInterpolationOffsetBits = 4;
  l.maxFramebufferWidth = l.maxFramebufferHeight = 16384; l.maxFramebufferLayers = 2048;
  VkSampleCountFlags sc = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT | VK_SAMPLE_COUNT_8_BIT;
  l.framebufferColorSampleCounts = l.framebufferDepthSampleCounts = l.framebufferStencilSampleCounts =
  l.framebufferNoAttachmentsSampleCounts = sc;
  l.maxColorAttachments = 8;
  l.sampledImageColorSampleCounts = l.sampledImageIntegerSampleCounts =
  l.sampledImageDepthSampleCounts = l.sampledImageStencilSampleCounts = sc;
  l.storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT; l.maxSampleMaskWords = 1;
  l.timestampComputeAndGraphics = VK_TRUE; l.timestampPeriod = 1.0f;
  l.maxClipDistances = l.maxCullDistances = l.maxCombinedClipAndCullDistances = 8;
  l.discreteQueuePriorities = 2;
  l.pointSizeRange[0] = 1; l.pointSizeRange[1] = 64; l.lineWidthRange[0] = 1; l.lineWidthRange[1] = 8;
  l.pointSizeGranularity = 0.125f; l.lineWidthGranularity = 0.125f;
  l.strictLines = VK_TRUE; l.standardSampleLocations = VK_TRUE;
  l.optimalBufferCopyOffsetAlignment = 1; l.optimalBufferCopyRowPitchAlignment = 1;
  l.nonCoherentAtomSize = 64;
}

const char* kInstanceExt[] = {"VK_KHR_surface", "VK_EXT_headless_surface", "VK_EXT_debug_utils",
                              "VK_KHR_get_physical_device_properties2",
                              // window-system surfaces: accepted and answered like a headless surface
                              "VK_KHR_xcb_surface", "VK_KHR_xlib_surface", "VK_KHR_wayland_surface",
                              "VK_KHR_external_memory_capabilities", "VK_KHR_external_semaphore_capabilities",
                              "VK_KHR_external_fence_capabilities", "VK_KHR_device_group_creation"};
// Device extensions: the ones promoted to core up to 1.3 plus extensions whose
// commands the trace records. Extensions whose work the trace cannot model
// (descriptor buffers/heaps, push descriptors, shader
// objects, pipeline libraries, host image copy, ...) are deliberately not
// offered, so programs that require them exit at startup instead of producing
// traces with holes.
const char* kDeviceExt[] = {"VK_KHR_swapchain", "VK_KHR_dynamic_rendering", "VK_KHR_synchronization2",
                            "VK_KHR_timeline_semaphore", "VK_KHR_maintenance4", "VK_KHR_buffer_device_address",
                            "VK_EXT_descriptor_indexing",
                            "VK_KHR_maintenance1", "VK_KHR_maintenance2", "VK_KHR_maintenance3",
                            "VK_KHR_create_renderpass2", "VK_KHR_depth_stencil_resolve", "VK_KHR_multiview",
                            "VK_KHR_spirv_1_4", "VK_KHR_shader_float_controls", "VK_KHR_shader_draw_parameters",
                            "VK_KHR_shader_non_semantic_info", "VK_KHR_copy_commands2",
                            "VK_KHR_format_feature_flags2", "VK_KHR_get_memory_requirements2",
                            "VK_KHR_dedicated_allocation", "VK_KHR_bind_memory2", "VK_KHR_image_format_list",
                            "VK_KHR_sampler_mirror_clamp_to_edge", "VK_KHR_driver_properties",
                            "VK_KHR_imageless_framebuffer", "VK_KHR_separate_depth_stencil_layouts",
                            "VK_KHR_uniform_buffer_standard_layout", "VK_KHR_shader_float16_int8",
                            "VK_KHR_16bit_storage", "VK_KHR_8bit_storage", "VK_KHR_storage_buffer_storage_class",
                            "VK_KHR_vulkan_memory_model", "VK_KHR_shader_subgroup_extended_types",
                            "VK_KHR_shader_terminate_invocation", "VK_KHR_zero_initialize_workgroup_memory",
                            "VK_KHR_shader_integer_dot_product", "VK_KHR_sampler_ycbcr_conversion",
                            "VK_KHR_relaxed_block_layout", "VK_KHR_variable_pointers",
                            "VK_EXT_scalar_block_layout", "VK_EXT_host_query_reset", "VK_EXT_sampler_filter_minmax",
                            "VK_EXT_separate_stencil_usage", "VK_EXT_shader_viewport_index_layer",
                            "VK_EXT_extended_dynamic_state", "VK_EXT_extended_dynamic_state2",
                            "VK_EXT_extended_dynamic_state3", "VK_EXT_vertex_input_dynamic_state",
                            "VK_EXT_inline_uniform_block", "VK_EXT_conditional_rendering",
                            "VK_EXT_conservative_rasterization", "VK_EXT_memory_budget",
                            "VK_EXT_shader_demote_to_helper_invocation", "VK_EXT_subgroup_size_control",
                            "VK_EXT_texel_buffer_alignment", "VK_EXT_texture_compression_astc_hdr",
                            "VK_EXT_4444_formats", "VK_EXT_ycbcr_2plane_444_formats",
                            "VK_EXT_pipeline_creation_feedback", "VK_EXT_pipeline_creation_cache_control",
                            "VK_EXT_private_data", "VK_EXT_tooling_info", "VK_EXT_image_robustness",
                            "VK_EXT_debug_marker",
                            // ray tracing, ray queries and mesh shading: modelled (README "Ray tracing
                            // and mesh shading"); compute indirect-count draws and dispatch base too
                            "VK_KHR_acceleration_structure", "VK_KHR_ray_tracing_pipeline", "VK_KHR_ray_query",
                            "VK_KHR_deferred_host_operations", "VK_KHR_pipeline_library",
                            "VK_KHR_ray_tracing_maintenance1", "VK_KHR_ray_tracing_position_fetch",
                            "VK_EXT_mesh_shader", "VK_KHR_draw_indirect_count", "VK_KHR_device_group",
                            "VK_EXT_ray_tracing_invocation_reorder"};

template <size_t N>
VkResult enum_ext(const char* (&names)[N], uint32_t* pCount, VkExtensionProperties* out) {
  if (!out) { *pCount = N; return VK_SUCCESS; }
  uint32_t n = std::min<uint32_t>(*pCount, N);
  for (uint32_t i = 0; i < n; i++) {
    memset(&out[i], 0, sizeof out[i]);
    snprintf(out[i].extensionName, VK_MAX_EXTENSION_NAME_SIZE, "%s", names[i]);
    out[i].specVersion = 1;
  }
  *pCount = n;
  return n < N ? VK_INCOMPLETE : VK_SUCCESS;
}

template <class T>
VkResult fill_array(const std::vector<T>& v, uint32_t* pCount, T* out) {
  if (!out) { *pCount = (uint32_t)v.size(); return VK_SUCCESS; }
  uint32_t n = std::min<uint32_t>(*pCount, (uint32_t)v.size());
  for (uint32_t i = 0; i < n; i++) out[i] = v[i];
  *pCount = n;
  return n < v.size() ? VK_INCOMPLETE : VK_SUCCESS;
}

std::string module_sha(const VkPipelineShaderStageCreateInfo& s) {
  if (s.module) { auto it = g_module_sha.find(H(s.module)); return it == g_module_sha.end() ? "" : it->second; }
  if (auto* ci = find_chain<VkShaderModuleCreateInfo>(s.pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)) {
    Sha1 h; h.update(ci->pCode, ci->codeSize); return h.hex();
  }
  return "";
}

void w_stages(W& w, uint32_t n, const VkPipelineShaderStageCreateInfo* st) {
  w.k("stages").ab();
  for (uint32_t i = 0; i < n; i++) {
    const auto& s = st[i];
    w.ob().k("stage").fl(flags_VkShaderStageFlagBits, s.stage);
    w.k("module").h(H(s.module)).k("sha1").str(module_sha(s)).k("entry").str(s.pName);
    if (s.pSpecializationInfo && s.pSpecializationInfo->mapEntryCount) {
      auto* si = s.pSpecializationInfo;
      w.k("specialization").ab();
      for (uint32_t j = 0; j < si->mapEntryCount; j++) {
        auto& e = si->pMapEntries[j];
        w.ob().k("id").u(e.constantID).k("value");
        if (e.offset + e.size <= si->dataSize) w.hex((const char*)si->pData + e.offset, e.size); else w.nul();
        w.oe();
      }
      w.ae();
    }
    w.oe();
  }
  w.ae();
}

void w_range(W& w, const VkImageSubresourceRange& r) {
  w.k("aspect").fl(flags_VkImageAspectFlagBits, r.aspectMask).k("base_mip").u(r.baseMipLevel)
   .k("level_count").u(r.levelCount).k("base_layer").u(r.baseArrayLayer).k("layer_count").u(r.layerCount);
}

void w_layers(W& w, const VkImageSubresourceLayers& r) {
  w.ob().k("aspect").fl(flags_VkImageAspectFlagBits, r.aspectMask).k("mip").u(r.mipLevel)
   .k("base_layer").u(r.baseArrayLayer).k("layer_count").u(r.layerCount).oe();
}

void w_image_barrier(W& w, uint64_t img, VkImageLayout o, VkImageLayout n, uint64_t ss, uint64_t sa,
                     uint64_t ds, uint64_t da, uint32_t sf, uint32_t df, const VkImageSubresourceRange& r) {
  w.ob().k("image").h(img).k("old_layout").en(name_VkImageLayout(o), o).k("new_layout").en(name_VkImageLayout(n), n);
  w.k("src_stage"); stages(w, ss); w.k("src_access"); access(w, sa);
  w.k("dst_stage"); stages(w, ds); w.k("dst_access"); access(w, da);
  w.k("src_family").family(sf).k("dst_family").family(df);
  w_range(w, r);
  w.oe();
}

void w_buffer_barrier(W& w, uint64_t buf, uint64_t ss, uint64_t sa, uint64_t ds, uint64_t da,
                      uint32_t sf, uint32_t df, VkDeviceSize off, VkDeviceSize size) {
  w.ob().k("buffer").h(buf);
  w.k("src_stage"); stages(w, ss); w.k("src_access"); access(w, sa);
  w.k("dst_stage"); stages(w, ds); w.k("dst_access"); access(w, da);
  w.k("src_family").family(sf).k("dst_family").family(df).k("offset").u(off);
  w.k("size"); if (size == VK_WHOLE_SIZE) w.str("whole"); else w.u(size);
  w.oe();
}

void w_memory_barrier(W& w, uint64_t ss, uint64_t sa, uint64_t ds, uint64_t da) {
  w.ob(); w.k("src_stage"); stages(w, ss); w.k("src_access"); access(w, sa);
  w.k("dst_stage"); stages(w, ds); w.k("dst_access"); access(w, da); w.oe();
}

// a VkClearValue is a union: record both readings (color as float4, depth/stencil)
void w_clear(W& w, const VkClearValue& c) {
  w.k("clear").ob().k("color").ab().d(c.color.float32[0]).d(c.color.float32[1]).d(c.color.float32[2])
   .d(c.color.float32[3]).ae().k("depth").d(c.depthStencil.depth).k("stencil").u(c.depthStencil.stencil).oe();
}

void w_attachment(W& w, const VkRenderingAttachmentInfo* a) {
  if (!a) { w.nul(); return; }
  w.ob().k("view").h(H(a->imageView)).k("layout").en(name_VkImageLayout(a->imageLayout), a->imageLayout)
   .k("load").en(name_VkAttachmentLoadOp(a->loadOp), a->loadOp)
   .k("store").en(name_VkAttachmentStoreOp(a->storeOp), a->storeOp)
   .k("resolve_view").h(H(a->resolveImageView))
   .k("resolve_layout").en(name_VkImageLayout(a->resolveImageLayout), a->resolveImageLayout);
  if (a->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) w_clear(w, a->clearValue);
  w.oe();
}

void w_sem_infos(W& w, const char* key, uint32_t n, const VkSemaphore* sems, const uint64_t* vals,
                 uint32_t nvals, const VkPipelineStageFlags* st) {
  w.k(key).ab();
  for (uint32_t i = 0; i < n; i++) {
    w.ob().k("semaphore").h(H(sems[i]));
    w.k("value"); if (vals && i < nvals) w.u(vals[i]); else w.nul();
    if (st) { w.k("stage"); stages(w, st[i]); }
    w.oe();
  }
  w.ae();
}

void w_sem_infos2(W& w, const char* key, uint32_t n, const VkSemaphoreSubmitInfo* s) {
  w.k(key).ab();
  for (uint32_t i = 0; i < n; i++) {
    w.ob().k("semaphore").h(H(s[i].semaphore)).k("value").u(s[i].value).k("stage");
    stages(w, s[i].stageMask); w.oe();
  }
  w.ae();
}

}  // namespace

extern "C" void covvk_unimplemented(const char* name) {
  ENTER;
  W w = rec(name); w.k("warning").str("unimplemented"); emit(w);
  if (g_warned.insert(name).second) fprintf(stderr, "covvk: warning: %s is not modelled (recorded only)\n", name);
}

extern "C" void covvk_unimplemented_cmd(const char* name, uint64_t cb) {
  ENTER;
  W w = rec(name); w.k("cb").h(cb).k("warning").str("unimplemented"); emit(w);
  if (g_warned.insert(name).second) fprintf(stderr, "covvk: warning: %s is not modelled (recorded only)\n", name);
}

extern "C" void covvk_record_cmd(const char* name, uint64_t cb) {
  ENTER;
  W w = rec(name); w.k("cb").h(cb); emit(w);
}

template <class T> T covvk_fake_handle(const char* kind) {
  std::lock_guard<std::recursive_mutex> lk(g_mu);
  return as<T>(mk(kind));
}

// ============================================================ entry points
extern "C" {

// ---------------------------------------------------------------- instance
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceVersion(uint32_t* v) {
  *v = VK_MAKE_API_VERSION(0, 1, 3, VK_HEADER_VERSION); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char* layer, uint32_t* n, VkExtensionProperties* p) {
  if (layer) { *n = 0; return VK_ERROR_LAYER_NOT_PRESENT; }
  return enum_ext(kInstanceExt, n, p);
}
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t* n, VkLayerProperties*) { *n = 0; return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice, const char* layer, uint32_t* n, VkExtensionProperties* p) {
  if (layer) { *n = 0; return VK_ERROR_LAYER_NOT_PRESENT; }
  return enum_ext(kDeviceExt, n, p);
}
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(VkPhysicalDevice, uint32_t* n, VkLayerProperties*) { *n = 0; return VK_SUCCESS; }

VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* ci, const VkAllocationCallbacks*, VkInstance* out) {
  ENTER;
  g_instance = mk("instance");
  *out = as<VkInstance>(g_instance);
  W w = rec("vkCreateInstance");
  w.k("instance").h(g_instance);
  if (ci->pApplicationInfo) {
    w.k("application").str(ci->pApplicationInfo->pApplicationName)
     .k("api_version").str(std::to_string(VK_API_VERSION_MAJOR(ci->pApplicationInfo->apiVersion)) + "." +
                           std::to_string(VK_API_VERSION_MINOR(ci->pApplicationInfo->apiVersion)));
  }
  w.k("layers").ab(); for (uint32_t i = 0; i < ci->enabledLayerCount; i++) w.str(ci->ppEnabledLayerNames[i]); w.ae();
  w.k("extensions").ab(); for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) w.str(ci->ppEnabledExtensionNames[i]); w.ae();
  emit(w);
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance, uint32_t* n, VkPhysicalDevice* p) {
  ENTER;
  if (!g_phys) g_phys = mk("physical_device");
  if (!p) { *n = 1; return VK_SUCCESS; }
  if (*n < 1) return VK_INCOMPLETE;
  *n = 1; p[0] = as<VkPhysicalDevice>(g_phys);
  W w = rec("vkEnumeratePhysicalDevices"); w.k("physical_devices").ab().h(g_phys).ae(); emit(w);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties* p) {
  memset(p, 0, sizeof *p);
  p->apiVersion = VK_MAKE_API_VERSION(0, 1, 3, VK_HEADER_VERSION);
  p->driverVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  p->vendorID = 0xC0DE; p->deviceID = 0x0001;
  p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  snprintf(p->deviceName, sizeof p->deviceName, "COVENANT model GPU (no hardware)");
  for (int i = 0; i < VK_UUID_SIZE; i++) p->pipelineCacheUUID[i] = (uint8_t)(0xC0 + i);
  fill_limits(p->limits);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice pd, VkPhysicalDeviceProperties2* p) {
  vkGetPhysicalDeviceProperties(pd, &p->properties);
  for (auto* s = (VkBaseOutStructure*)p->pNext; s; s = s->pNext) {
    switch (s->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES: {
        auto* q = (VkPhysicalDeviceVulkan11Properties*)s;
        for (int i = 0; i < VK_UUID_SIZE; i++) { q->deviceUUID[i] = (uint8_t)i; q->driverUUID[i] = (uint8_t)(0x80 + i); }
        q->subgroupSize = 32; q->subgroupSupportedStages = VK_SHADER_STAGE_ALL;
        q->subgroupSupportedOperations = 0xFF; q->maxMultiviewViewCount = 6;
        q->maxMultiviewInstanceIndex = (1u << 27) - 1; q->maxPerSetDescriptors = 1u << 20;
        q->maxMemoryAllocationSize = 1ull << 32; q->pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES: {
        auto* q = (VkPhysicalDeviceVulkan12Properties*)s;
        q->driverID = VK_DRIVER_ID_MESA_LLVMPIPE;
        snprintf(q->driverName, VK_MAX_DRIVER_NAME_SIZE, "covvk");
        snprintf(q->driverInfo, VK_MAX_DRIVER_INFO_SIZE, "COVENANT model driver");
        q->conformanceVersion = {1, 3, 0, 0};
        q->maxTimelineSemaphoreValueDifference = UINT64_MAX;
        q->supportedDepthResolveModes = q->supportedStencilResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
        q->maxUpdateAfterBindDescriptorsInAllPools = 1u << 20;
        q->maxPerStageDescriptorUpdateAfterBindSamplers = q->maxPerStageDescriptorUpdateAfterBindSampledImages =
        q->maxPerStageDescriptorUpdateAfterBindStorageBuffers = q->maxPerStageDescriptorUpdateAfterBindStorageImages =
        q->maxPerStageDescriptorUpdateAfterBindUniformBuffers = q->maxPerStageUpdateAfterBindResources = 1u << 20;
        q->maxDescriptorSetUpdateAfterBindSamplers = q->maxDescriptorSetUpdateAfterBindSampledImages =
        q->maxDescriptorSetUpdateAfterBindStorageBuffers = q->maxDescriptorSetUpdateAfterBindStorageImages =
        q->maxDescriptorSetUpdateAfterBindUniformBuffers = 1u << 20;
        q->framebufferIntegerColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES: {
        auto* q = (VkPhysicalDeviceVulkan13Properties*)s;
        q->minSubgroupSize = q->maxSubgroupSize = 32; q->maxComputeWorkgroupSubgroups = 32;
        q->requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;
        q->maxInlineUniformBlockSize = 256; q->maxInlineUniformTotalSize = 1u << 16;
        q->maxBufferSize = 1ull << 31;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES: {
        auto* q = (VkPhysicalDeviceDriverProperties*)s;
        q->driverID = VK_DRIVER_ID_MESA_LLVMPIPE;
        snprintf(q->driverName, VK_MAX_DRIVER_NAME_SIZE, "covvk");
        snprintf(q->driverInfo, VK_MAX_DRIVER_INFO_SIZE, "COVENANT model driver");
        q->conformanceVersion = {1, 3, 0, 0};
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES:
        ((VkPhysicalDeviceTimelineSemaphoreProperties*)s)->maxTimelineSemaphoreValueDifference = UINT64_MAX; break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES: {  // filled in (zeroed limits break applications)
        auto* q = (VkPhysicalDeviceMaintenance3Properties*)s;
        q->maxPerSetDescriptors = 1u << 20; q->maxMemoryAllocationSize = 1ull << 32;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES:
        ((VkPhysicalDeviceMaintenance4Properties*)s)->maxBufferSize = 1ull << 31; break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR: {
        auto* q = (VkPhysicalDeviceAccelerationStructurePropertiesKHR*)s;
        q->maxGeometryCount = 1ull << 24; q->maxInstanceCount = 1ull << 24; q->maxPrimitiveCount = 1ull << 29;
        q->maxPerStageDescriptorAccelerationStructures = q->maxPerStageDescriptorUpdateAfterBindAccelerationStructures =
        q->maxDescriptorSetAccelerationStructures = q->maxDescriptorSetUpdateAfterBindAccelerationStructures = 1u << 16;
        q->minAccelerationStructureScratchOffsetAlignment = 128;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR: {
        auto* q = (VkPhysicalDeviceRayTracingPipelinePropertiesKHR*)s;
        q->shaderGroupHandleSize = 32; q->maxRayRecursionDepth = 31; q->maxShaderGroupStride = 4096;
        q->shaderGroupBaseAlignment = 64; q->shaderGroupHandleCaptureReplaySize = 32;
        q->maxRayDispatchInvocationCount = 1u << 30; q->shaderGroupHandleAlignment = 32; q->maxRayHitAttributeSize = 32;
        break;
      }
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT: {
        auto* q = (VkPhysicalDeviceMeshShaderPropertiesEXT*)s;
        q->maxTaskWorkGroupTotalCount = q->maxMeshWorkGroupTotalCount = 1u << 22;
        for (int i = 0; i < 3; i++) {
          q->maxTaskWorkGroupCount[i] = q->maxMeshWorkGroupCount[i] = 65535;
          q->maxTaskWorkGroupSize[i] = q->maxMeshWorkGroupSize[i] = 128;
        }
        q->maxTaskWorkGroupInvocations = q->maxMeshWorkGroupInvocations = 128;
        q->maxTaskPayloadSize = 16384; q->maxTaskSharedMemorySize = q->maxMeshSharedMemorySize = 32768;
        q->maxTaskPayloadAndSharedMemorySize = q->maxMeshPayloadAndSharedMemorySize = 32768;
        q->maxMeshOutputMemorySize = q->maxMeshPayloadAndOutputMemorySize = 48128;
        q->maxMeshOutputComponents = 128; q->maxMeshOutputVertices = 256; q->maxMeshOutputPrimitives = 256;
        q->maxMeshOutputLayers = 8; q->maxMeshMultiviewViewCount = 4;
        q->meshOutputPerVertexGranularity = q->meshOutputPerPrimitiveGranularity = 32;
        q->maxPreferredTaskWorkGroupInvocations = q->maxPreferredMeshWorkGroupInvocations = 32;
        break;
      }
      default: break;
    }
  }
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(VkPhysicalDevice, VkPhysicalDeviceFeatures* f) {
  VkBool32* b = (VkBool32*)f;
  for (size_t i = 0; i < sizeof *f / sizeof(VkBool32); i++) b[i] = VK_TRUE;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice pd, VkPhysicalDeviceFeatures2* f) {
  vkGetPhysicalDeviceFeatures(pd, &f->features);
  for (auto* s = (VkBaseOutStructure*)f->pNext; s; s = s->pNext) {
    switch (s->sType) {
#define B(ST, T) case ST: fill_bools(s, sizeof(T)); break;
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, VkPhysicalDeviceVulkan12Features)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, VkPhysicalDeviceVulkan13Features)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, VkPhysicalDeviceDynamicRenderingFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES, VkPhysicalDeviceSynchronization2Features)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, VkPhysicalDeviceTimelineSemaphoreFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, VkPhysicalDeviceBufferDeviceAddressFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES, VkPhysicalDeviceDescriptorIndexingFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES, VkPhysicalDeviceScalarBlockLayoutFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, VkPhysicalDeviceMaintenance4Features)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES, VkPhysicalDeviceShaderDrawParametersFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, VkPhysicalDevice16BitStorageFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, VkPhysicalDevice8BitStorageFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, VkPhysicalDeviceShaderFloat16Int8Features)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES, VkPhysicalDeviceHostQueryResetFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES, VkPhysicalDeviceImagelessFramebufferFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES, VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES, VkPhysicalDeviceUniformBufferStandardLayoutFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, VkPhysicalDeviceVulkanMemoryModelFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES, VkPhysicalDeviceMultiviewFeatures)
      B(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES, VkPhysicalDeviceSamplerYcbcrConversionFeatures)
#undef B
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
        fill_bools(s, sizeof(VkPhysicalDeviceVulkan11Features));
        ((VkPhysicalDeviceVulkan11Features*)s)->protectedMemory = VK_FALSE;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES: break;   // stays unsupported
      default:
        // any other all-VkBool32 feature struct (table generated from vk.xml)
        for (auto* e = covvk_feature_structs; e->size; e++)
          if (e->st == s->sType) { fill_bools(s, e->size); break; }
        break;
    }
  }
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, uint32_t* n, VkQueueFamilyProperties* p) {
  if (!p) { *n = 2; return; }
  VkQueueFamilyProperties f[2] = {};
  f[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
  f[0].queueCount = 2; f[0].timestampValidBits = 64; f[0].minImageTransferGranularity = {1, 1, 1};
  f[1].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
  f[1].queueCount = 1; f[1].timestampValidBits = 64; f[1].minImageTransferGranularity = {1, 1, 1};
  *n = std::min(*n, 2u);
  for (uint32_t i = 0; i < *n; i++) p[i] = f[i];
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice pd, uint32_t* n, VkQueueFamilyProperties2* p) {
  if (!p) { *n = 2; return; }
  VkQueueFamilyProperties f[2]; uint32_t m = 2;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &m, f);
  *n = std::min(*n, 2u);
  for (uint32_t i = 0; i < *n; i++) p[i].queueFamilyProperties = f[i];
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* p) {
  memset(p, 0, sizeof *p);
  p->memoryHeapCount = 2;
  p->memoryHeaps[0] = {8ull << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
  p->memoryHeaps[1] = {16ull << 30, 0};
  p->memoryTypeCount = 4;
  for (int i = 0; i < 4; i++) p->memoryTypes[i] = {kMemTypes[i], kMemHeap[i]};
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2(VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2* p) {
  vkGetPhysicalDeviceMemoryProperties(pd, &p->memoryProperties);
  // VK_EXT_memory_budget is offered, so a chained VkPhysicalDeviceMemoryBudgetPropertiesEXT
  // must be filled: heapBudget is non-zero for every heap below memoryHeapCount and at most the
  // heap's size, heapUsage is this process's allocated bytes (memory.adoc:1079-1112). Leaving it
  // untouched left heapBudget 0, and an application that meters allocations by the budget (VSG,
  // VMA with the extension enabled) then sees no memory at all.
  for (auto* s = (VkBaseOutStructure*)p->pNext; s; s = s->pNext) {
    if (s->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT) continue;
    auto* b = (VkPhysicalDeviceMemoryBudgetPropertiesEXT*)s;
    memset(b->heapBudget, 0, sizeof b->heapBudget);
    memset(b->heapUsage, 0, sizeof b->heapUsage);
    for (uint32_t i = 0; i < p->memoryProperties.memoryHeapCount; i++)
      b->heapBudget[i] = p->memoryProperties.memoryHeaps[i].size;
    for (auto& kv : g_mem) {
      uint32_t t = kv.second.type;
      if (t < 4 && kMemHeap[t] < p->memoryProperties.memoryHeapCount) b->heapUsage[kMemHeap[t]] += kv.second.size;
    }
  }
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice, VkFormat f, VkFormatProperties* p) {
  if (f == VK_FORMAT_UNDEFINED) { *p = {}; return; }
  bool depth = f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D24_UNORM_S8_UINT ||
               f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_S8_UINT ||
               f == VK_FORMAT_D16_UNORM_S8_UINT;
  VkFormatFeatureFlags all = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT;
  VkFormatFeatureFlags opt = all | (depth ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
      : (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
         VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT));
  p->optimalTilingFeatures = opt;
  p->linearTilingFeatures = depth ? 0 : opt;
  p->bufferFeatures = depth ? 0 : (VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT | VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT |
                                   VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT);
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2(VkPhysicalDevice pd, VkFormat f, VkFormatProperties2* p) {
  vkGetPhysicalDeviceFormatProperties(pd, f, &p->formatProperties);
  for (auto* s = (VkBaseOutStructure*)p->pNext; s; s = s->pNext)
    if (s->sType == VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3) {
      auto* q = (VkFormatProperties3*)s;
      q->linearTilingFeatures = p->formatProperties.linearTilingFeatures;
      q->optimalTilingFeatures = p->formatProperties.optimalTilingFeatures;
      q->bufferFeatures = p->formatProperties.bufferFeatures;
    }
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(VkPhysicalDevice, VkFormat, VkImageType t, VkImageTiling,
                                                                        VkImageUsageFlags, VkImageCreateFlags, VkImageFormatProperties* p) {
  p->maxExtent = {16384, t == VK_IMAGE_TYPE_1D ? 1u : 16384u, t == VK_IMAGE_TYPE_3D ? 2048u : 1u};
  p->maxMipLevels = 15; p->maxArrayLayers = 2048;
  p->sampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT | VK_SAMPLE_COUNT_8_BIT;
  p->maxResourceSize = 1ull << 32;
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties2(VkPhysicalDevice pd, const VkPhysicalDeviceImageFormatInfo2* i,
                                                                         VkImageFormatProperties2* p) {
  return vkGetPhysicalDeviceImageFormatProperties(pd, i->format, i->type, i->tiling, i->usage, i->flags, &p->imageFormatProperties);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance, const char* name);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice, const char* name);

VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance inst, const VkAllocationCallbacks*) {
  ENTER; W w = rec("vkDestroyInstance"); w.k("handle").h(H(inst)); emit(w);
}

// ---------------------------------------------------------------- device
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice, const VkDeviceCreateInfo* ci, const VkAllocationCallbacks*, VkDevice* out) {
  ENTER;
  g_device = mk("device");
  *out = as<VkDevice>(g_device);
  W w = rec("vkCreateDevice");
  w.k("device").h(g_device).k("queues").ab();
  for (uint32_t i = 0; i < ci->queueCreateInfoCount; i++)
    w.ob().k("family").u(ci->pQueueCreateInfos[i].queueFamilyIndex).k("count").u(ci->pQueueCreateInfos[i].queueCount).oe();
  w.ae().k("extensions").ab();
  for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) w.str(ci->ppEnabledExtensionNames[i]);
  w.ae();
  w.k("features").ab();
  if (find_chain<VkPhysicalDeviceVulkan13Features>(ci->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)) {
    auto* f = find_chain<VkPhysicalDeviceVulkan13Features>(ci->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
    if (f->dynamicRendering) w.str("dynamicRendering");
    if (f->synchronization2) w.str("synchronization2");
  }
  if (auto* f = find_chain<VkPhysicalDeviceVulkan12Features>(ci->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES))
    if (f->timelineSemaphore) w.str("timelineSemaphore");
  w.ae();
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice d, const VkAllocationCallbacks*) {
  ENTER; W w = rec("vkDestroyDevice"); w.k("handle").h(H(d)); emit(w);
}

static uint64_t get_queue(uint32_t fam, uint32_t idx) {
  auto key = std::make_pair(fam, idx);
  auto it = g_queues.find(key);
  if (it != g_queues.end()) return it->second;
  uint64_t q = mk("queue");
  g_queues[key] = q;
  return q;
}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice, uint32_t fam, uint32_t idx, VkQueue* out) {
  ENTER;
  uint64_t q = get_queue(fam, idx);
  *out = as<VkQueue>(q);
  W w = rec("vkGetDeviceQueue"); w.k("queue").h(q).k("family").u(fam).k("index").u(idx); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue2(VkDevice d, const VkDeviceQueueInfo2* i, VkQueue* out) {
  vkGetDeviceQueue(d, i->queueFamilyIndex, i->queueIndex, out);
}
VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice d) {
  ENTER; W w = rec("vkDeviceWaitIdle"); w.k("device").h(H(d)); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue q) {
  ENTER; W w = rec("vkQueueWaitIdle"); w.k("queue").h(H(q)); emit(w); return VK_SUCCESS;
}

// ---------------------------------------------------------------- debug utils
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(VkInstance, const VkDebugUtilsMessengerCreateInfoEXT*,
                                                              const VkAllocationCallbacks*, VkDebugUtilsMessengerEXT* out) {
  ENTER; uint64_t h = mk("debug_messenger"); *out = as<VkDebugUtilsMessengerEXT>(h);
  W w = rec("vkCreateDebugUtilsMessengerEXT"); w.k("messenger").h(h); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkSetDebugUtilsObjectNameEXT(VkDevice, const VkDebugUtilsObjectNameInfoEXT* i) {
  ENTER;
  W w = rec("vkSetDebugUtilsObjectNameEXT");
  w.k("object").h(i->objectHandle).k("object_type").en(name_VkObjectType(i->objectType), i->objectType).k("name").str(i->pObjectName);
  emit(w); return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkCmdBeginDebugUtilsLabelEXT(VkCommandBuffer cb, const VkDebugUtilsLabelEXT* l) {
  ENTER; W w = rec("vkCmdBeginDebugUtilsLabelEXT"); w.k("cb").h(H(cb)).k("label").str(l->pLabelName); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdEndDebugUtilsLabelEXT(VkCommandBuffer cb) {
  ENTER; W w = rec("vkCmdEndDebugUtilsLabelEXT"); w.k("cb").h(H(cb)); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdInsertDebugUtilsLabelEXT(VkCommandBuffer cb, const VkDebugUtilsLabelEXT* l) {
  ENTER; W w = rec("vkCmdInsertDebugUtilsLabelEXT"); w.k("cb").h(H(cb)).k("label").str(l->pLabelName); emit(w);
}

// ---------------------------------------------------------------- surface / swapchain
VKAPI_ATTR VkResult VKAPI_CALL vkCreateHeadlessSurfaceEXT(VkInstance, const VkHeadlessSurfaceCreateInfoEXT*,
                                                          const VkAllocationCallbacks*, VkSurfaceKHR* out) {
  ENTER; uint64_t h = mk("surface"); *out = as<VkSurfaceKHR>(h);
  W w = rec("vkCreateHeadlessSurfaceEXT"); w.k("surface").h(h); emit(w); return VK_SUCCESS;
}
// Window-system surfaces (xcb, xlib, wayland). The driver does not include the
// platform headers: the create-info structs all start with sType/pNext/flags and
// the window handles are only recorded as present. The surface behaves exactly
// like a headless one (extent from COVENANT_SURFACE_EXTENT).
static VkResult platform_surface(const char* fn, VkSurfaceKHR* out) {
  ENTER; uint64_t h = mk("surface"); *out = as<VkSurfaceKHR>(h);
  W w = rec(fn); w.k("surface").h(h); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateXcbSurfaceKHR(VkInstance, const void*, const VkAllocationCallbacks*, VkSurfaceKHR* out) {
  return platform_surface("vkCreateXcbSurfaceKHR", out);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateXlibSurfaceKHR(VkInstance, const void*, const VkAllocationCallbacks*, VkSurfaceKHR* out) {
  return platform_surface("vkCreateXlibSurfaceKHR", out);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateWaylandSurfaceKHR(VkInstance, const void*, const VkAllocationCallbacks*, VkSurfaceKHR* out) {
  return platform_surface("vkCreateWaylandSurfaceKHR", out);
}
VKAPI_ATTR VkBool32 VKAPI_CALL vkGetPhysicalDeviceXcbPresentationSupportKHR(VkPhysicalDevice, uint32_t fam, void*, uint32_t) {
  return fam == 0 ? VK_TRUE : VK_FALSE;
}
VKAPI_ATTR VkBool32 VKAPI_CALL vkGetPhysicalDeviceXlibPresentationSupportKHR(VkPhysicalDevice, uint32_t fam, void*, unsigned long) {
  return fam == 0 ? VK_TRUE : VK_FALSE;
}
VKAPI_ATTR VkBool32 VKAPI_CALL vkGetPhysicalDeviceWaylandPresentationSupportKHR(VkPhysicalDevice, uint32_t fam, void*) {
  return fam == 0 ? VK_TRUE : VK_FALSE;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice, uint32_t fam, VkSurfaceKHR, VkBool32* s) {
  *s = fam == 0 ? VK_TRUE : VK_FALSE; return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR* c) {
  auto e = surface_extent();
  c->minImageCount = 2; c->maxImageCount = 8;
  c->currentExtent = {e.first, e.second};
  c->minImageExtent = {1, 1}; c->maxImageExtent = {16384, 16384};
  c->maxImageArrayLayers = 1;
  c->supportedTransforms = c->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  c->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  c->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice, VkSurfaceKHR, uint32_t* n, VkSurfaceFormatKHR* f) {
  std::vector<VkSurfaceFormatKHR> v = {
    {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
    {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
    {VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
    {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
    {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};
  return fill_array(v, n, f);
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice, VkSurfaceKHR, uint32_t* n, VkPresentModeKHR* m) {
  std::vector<VkPresentModeKHR> v = {VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAILBOX_KHR,
                                     VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_RELAXED_KHR};
  return fill_array(v, n, m);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(VkDevice, const VkSwapchainCreateInfoKHR* ci, const VkAllocationCallbacks*,
                                                    VkSwapchainKHR* out) {
  ENTER;
  uint64_t h = mk("swapchain"); *out = as<VkSwapchainKHR>(h);
  Swap& s = g_swap[h];
  uint32_t count = std::max(ci->minImageCount, 3u);
  for (uint32_t i = 0; i < count; i++) {
    uint64_t img = mk("image");
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = ci->imageFormat;
    ii.extent = {ci->imageExtent.width, ci->imageExtent.height, 1};
    ii.mipLevels = 1; ii.arrayLayers = ci->imageArrayLayers; ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.usage = ci->imageUsage;
    g_image[img] = ii;
    s.images.push_back(img);
  }
  W w = rec("vkCreateSwapchainKHR");
  w.k("swapchain").h(h).k("surface").h(H(ci->surface))
   .k("format").en(name_VkFormat(ci->imageFormat), ci->imageFormat)
   .k("color_space").en(name_VkColorSpaceKHR(ci->imageColorSpace), ci->imageColorSpace)
   .k("extent").ab().u(ci->imageExtent.width).u(ci->imageExtent.height).ae()
   .k("min_image_count").u(ci->minImageCount).k("image_count").u(count)
   .k("surface_min_image_count").u(2)
   .k("usage").fl(flags_VkImageUsageFlagBits, ci->imageUsage)
   .k("sharing").en(name_VkSharingMode(ci->imageSharingMode), ci->imageSharingMode)
   .k("present_mode").en(name_VkPresentModeKHR(ci->presentMode), ci->presentMode)
   .k("old_swapchain").h(H(ci->oldSwapchain));
  w.k("images").ab(); for (auto i : s.images) w.h(i); w.ae();
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(VkDevice, VkSwapchainKHR sc, uint32_t* n, VkImage* imgs) {
  ENTER;
  Swap& s = g_swap[H(sc)];
  if (!imgs) { *n = (uint32_t)s.images.size(); return VK_SUCCESS; }
  uint32_t m = std::min<uint32_t>(*n, (uint32_t)s.images.size());
  for (uint32_t i = 0; i < m; i++) imgs[i] = as<VkImage>(s.images[i]);
  *n = m;
  W w = rec("vkGetSwapchainImagesKHR"); w.k("swapchain").h(H(sc)).k("images").ab();
  for (uint32_t i = 0; i < m; i++) w.h(s.images[i]);
  w.ae(); emit(w);
  return m < s.images.size() ? VK_INCOMPLETE : VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImageKHR(VkDevice, VkSwapchainKHR sc, uint64_t timeout, VkSemaphore sem,
                                                     VkFence fence, uint32_t* idx) {
  ENTER;
  Swap& s = g_swap[H(sc)];
  uint32_t n = (uint32_t)std::max<size_t>(1, s.images.size());
  *idx = s.next % n; s.next = (s.next + 1) % n;
  W w = rec("vkAcquireNextImageKHR");
  w.k("swapchain").h(H(sc)).k("semaphore").h(H(sem)).k("fence").h(H(fence)).k("timeout");
  if (timeout == UINT64_MAX) w.str("infinite"); else w.u(timeout);
  w.k("image_index").u(*idx);
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImage2KHR(VkDevice d, const VkAcquireNextImageInfoKHR* i, uint32_t* idx) {
  return vkAcquireNextImageKHR(d, i->swapchain, i->timeout, i->semaphore, i->fence, idx);
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue q, const VkPresentInfoKHR* p) {
  ENTER;
  W w = rec("vkQueuePresentKHR");
  w.k("queue").h(H(q));
  w_sem_infos(w, "wait", p->waitSemaphoreCount, p->pWaitSemaphores, nullptr, 0, nullptr);
  w.k("swapchains").ab();
  for (uint32_t i = 0; i < p->swapchainCount; i++) {
    w.ob().k("swapchain").h(H(p->pSwapchains[i])).k("image_index").u(p->pImageIndices[i]).oe();
    if (p->pResults) p->pResults[i] = VK_SUCCESS;
  }
  w.ae(); emit(w);
  // harness knob: COVENANT_MAX_PRESENTS=N ends the process right after the N-th present
  // is recorded (the trace is closed first), for programs without a frame-count option
  static long max_presents = [] { const char* e = getenv("COVENANT_MAX_PRESENTS"); return e ? atol(e) : 0L; }();
  static long presents = 0;
  if (max_presents > 0 && ++presents >= max_presents) {
    W e = rec("covenant_max_presents"); e.k("presents").u(presents); emit(e);
    finish();
    _exit(0);
  }
  return VK_SUCCESS;
}

// ---------------------------------------------------------------- memory, buffers, images
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks*, VkDeviceMemory* out) {
  ENTER;
  uint64_t h = mk("memory"); *out = as<VkDeviceMemory>(h);
  MemObj& m = g_mem[h]; m.size = ai->allocationSize; m.type = ai->memoryTypeIndex;
  W w = rec("vkAllocateMemory");
  w.k("memory").h(h).k("size").u(ai->allocationSize).k("type_index").u(ai->memoryTypeIndex)
   .k("properties").fl(flags_VkMemoryPropertyFlagBits, ai->memoryTypeIndex < 4 ? kMemTypes[ai->memoryTypeIndex] : 0);
  if (auto* d = find_chain<VkMemoryDedicatedAllocateInfo>(ai->pNext, VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO))
    w.k("dedicated_image").h(H(d->image)).k("dedicated_buffer").h(H(d->buffer));
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice, VkDeviceMemory mem, const VkAllocationCallbacks*) {
  ENTER;
  if (!mem) return;
  auto it = g_mem.find(H(mem));
  if (it != g_mem.end()) {
    if (it->second.region >= 0) {
      Region& r = g_reg[it->second.region];
      if (r.base.load()) mprotect(r.base.load(), r.len, PROT_READ | PROT_WRITE);   // no faults from here on
      r.base.store(nullptr);
      delete[] (uint8_t*)r.bits; r.bits = nullptr;
      delete[] r.shadow; r.shadow = nullptr;
    }
    if (it->second.base) munmap(it->second.base, it->second.len);
    g_mem.erase(it);
  }
  W w = rec("vkFreeMemory"); w.k("handle").h(H(mem)); emit(w);
}
VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice, VkDeviceMemory mem, VkDeviceSize off, VkDeviceSize size, VkMemoryMapFlags, void** pp) {
  ENTER;
  MemObj& m = g_mem[H(mem)];
  if (!m.base) {
    m.len = ((size_t)m.size + g_page - 1) / g_page * g_page;
    if (!m.len) m.len = g_page;
    void* p = mmap(nullptr, m.len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return VK_ERROR_MEMORY_MAP_FAILED;
    m.base = (char*)p;
    if (g_track) {
      int slot = -1;
      for (int i = 0; i < g_nreg.load(); i++) if (!g_reg[i].base.load()) { slot = i; break; }
      if (slot < 0 && g_nreg.load() < MAXREG) slot = g_nreg.fetch_add(1);
      if (slot >= 0) {
        Region& r = g_reg[slot];
        r.len = m.len; r.mem = H(mem); r.dirty = 0;
        r.bits = new uint8_t[m.len / g_page]();
        r.shadow = new uint8_t[m.len]();     // anonymous mappings start zeroed
        r.base.store(m.base);
        m.region = slot;
        if (!g_handler) {
          struct sigaction sa = {};
          sa.sa_sigaction = segv; sa.sa_flags = SA_SIGINFO | SA_RESTART | SA_NODEFER;
          sigemptyset(&sa.sa_mask);
          sigaction(SIGSEGV, &sa, &g_old_segv);
          g_handler = true;
        }
      }
    }
  }
  if (m.region >= 0) mprotect(m.base, m.len, PROT_READ);
  *pp = m.base + off;
  W w = rec("vkMapMemory");
  w.k("memory").h(H(mem)).k("offset").u(off).k("size");
  if (size == VK_WHOLE_SIZE) w.str("whole"); else w.u(size);
  w.k("tracked").b(m.region >= 0);
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice, VkDeviceMemory mem) {
  ENTER; W w = rec("vkUnmapMemory"); w.k("memory").h(H(mem)); emit(w);
}
VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(VkDevice, uint32_t n, const VkMappedMemoryRange* r) {
  ENTER; W w = rec("vkFlushMappedMemoryRanges"); w.k("ranges").ab();
  for (uint32_t i = 0; i < n; i++) w.ob().k("memory").h(H(r[i].memory)).k("offset").u(r[i].offset).oe();
  w.ae(); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkInvalidateMappedMemoryRanges(VkDevice, uint32_t n, const VkMappedMemoryRange* r) {
  ENTER; W w = rec("vkInvalidateMappedMemoryRanges"); w.k("ranges").ab();
  for (uint32_t i = 0; i < n; i++) w.ob().k("memory").h(H(r[i].memory)).k("offset").u(r[i].offset).oe();
  w.ae(); emit(w); return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice, const VkBufferCreateInfo* ci, const VkAllocationCallbacks*, VkBuffer* out) {
  ENTER;
  uint64_t h = mk("buffer"); *out = as<VkBuffer>(h);
  g_bufsize[h] = ci->size;
  W w = rec("vkCreateBuffer");
  w.k("buffer").h(h).k("size").u(ci->size).k("usage").fl(flags_VkBufferUsageFlagBits, ci->usage)
   .k("sharing").en(name_VkSharingMode(ci->sharingMode), ci->sharingMode);
  emit(w);
  return VK_SUCCESS;
}
static void buf_reqs(VkDeviceSize size, VkMemoryRequirements* r) {
  r->size = (size + 255) & ~VkDeviceSize(255); r->alignment = 256; r->memoryTypeBits = 0xF;
}
VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice, VkBuffer b, VkMemoryRequirements* r) {
  std::lock_guard<std::recursive_mutex> lk(g_mu); buf_reqs(g_bufsize[H(b)], r);
}
VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements2(VkDevice d, const VkBufferMemoryRequirementsInfo2* i, VkMemoryRequirements2* r) {
  vkGetBufferMemoryRequirements(d, i->buffer, &r->memoryRequirements);
}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceBufferMemoryRequirements(VkDevice, const VkDeviceBufferMemoryRequirements* i, VkMemoryRequirements2* r) {
  buf_reqs(i->pCreateInfo->size, &r->memoryRequirements);
}
static void img_reqs(const VkImageCreateInfo& ci, VkMemoryRequirements* r) {
  r->size = image_bytes(ci); r->alignment = 4096;
  r->memoryTypeBits = ci.tiling == VK_IMAGE_TILING_LINEAR ? 0xF : 0x9;
}
VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(VkDevice, VkImage i, VkMemoryRequirements* r) {
  std::lock_guard<std::recursive_mutex> lk(g_mu); img_reqs(g_image[H(i)], r);
}
VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements2(VkDevice d, const VkImageMemoryRequirementsInfo2* i, VkMemoryRequirements2* r) {
  vkGetImageMemoryRequirements(d, i->image, &r->memoryRequirements);
}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceImageMemoryRequirements(VkDevice, const VkDeviceImageMemoryRequirements* i, VkMemoryRequirements2* r) {
  img_reqs(*i->pCreateInfo, &r->memoryRequirements);
}
VKAPI_ATTR void VKAPI_CALL vkGetImageSubresourceLayout(VkDevice, VkImage i, const VkImageSubresource*, VkSubresourceLayout* l) {
  std::lock_guard<std::recursive_mutex> lk(g_mu);
  auto& ci = g_image[H(i)];
  l->offset = 0; l->rowPitch = (VkDeviceSize)ci.extent.width * format_size(ci.format);
  l->size = image_bytes(ci); l->arrayPitch = l->depthPitch = l->rowPitch * ci.extent.height;
}
// pure query: the fake device has no tiling constraint (apps divide by this)
VKAPI_ATTR void VKAPI_CALL vkGetRenderAreaGranularity(VkDevice, VkRenderPass, VkExtent2D* g) {
  g->width = 1; g->height = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice, VkBuffer b, VkDeviceMemory m, VkDeviceSize off) {
  ENTER; g_bufbind[H(b)] = {H(m), off};
  W w = rec("vkBindBufferMemory"); w.k("buffer").h(H(b)).k("memory").h(H(m)).k("offset").u(off);
  w.k("size").u(g_bufsize[H(b)]); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory2(VkDevice d, uint32_t n, const VkBindBufferMemoryInfo* i) {
  for (uint32_t k = 0; k < n; k++) vkBindBufferMemory(d, i[k].buffer, i[k].memory, i[k].memoryOffset);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice, VkImage im, VkDeviceMemory m, VkDeviceSize off) {
  ENTER; W w = rec("vkBindImageMemory"); w.k("image").h(H(im)).k("memory").h(H(m)).k("offset").u(off);
  w.k("size").u(image_bytes(g_image[H(im)])); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory2(VkDevice d, uint32_t n, const VkBindImageMemoryInfo* i) {
  for (uint32_t k = 0; k < n; k++) vkBindImageMemory(d, i[k].image, i[k].memory, i[k].memoryOffset);
  return VK_SUCCESS;
}
VKAPI_ATTR VkDeviceAddress VKAPI_CALL vkGetBufferDeviceAddress(VkDevice, const VkBufferDeviceAddressInfo* i) {
  ENTER;
  VkDeviceAddress a = buf_addr(H(i->buffer));     // stable: the same buffer always has one address
  W w = rec("vkGetBufferDeviceAddress"); w.k("buffer").h(H(i->buffer)).k("address").u(a); emit(w);
  return a;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(VkDevice, const VkImageCreateInfo* ci, const VkAllocationCallbacks*, VkImage* out) {
  ENTER;
  uint64_t h = mk("image"); *out = as<VkImage>(h);
  g_image[h] = *ci; g_image[h].pNext = nullptr; g_image[h].pQueueFamilyIndices = nullptr;
  W w = rec("vkCreateImage");
  w.k("image").h(h).k("type").en(name_VkImageType(ci->imageType), ci->imageType)
   .k("format").en(name_VkFormat(ci->format), ci->format)
   .k("extent").ab().u(ci->extent.width).u(ci->extent.height).u(ci->extent.depth).ae()
   .k("mip_levels").u(ci->mipLevels).k("array_layers").u(ci->arrayLayers)
   .k("samples").u(ci->samples).k("tiling").en(name_VkImageTiling(ci->tiling), ci->tiling)
   .k("usage").fl(flags_VkImageUsageFlagBits, ci->usage)
   .k("sharing").en(name_VkSharingMode(ci->sharingMode), ci->sharingMode)
   .k("initial_layout").en(name_VkImageLayout(ci->initialLayout), ci->initialLayout);
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(VkDevice, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks*, VkImageView* out) {
  ENTER;
  uint64_t h = mk("image_view"); *out = as<VkImageView>(h);
  W w = rec("vkCreateImageView");
  w.k("view").h(h).k("image").h(H(ci->image)).k("format").en(name_VkFormat(ci->format), ci->format)
   .k("view_type").en(name_VkImageViewType(ci->viewType), ci->viewType);
  w_range(w, ci->subresourceRange);
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateBufferView(VkDevice, const VkBufferViewCreateInfo* ci, const VkAllocationCallbacks*, VkBufferView* out) {
  ENTER;
  uint64_t h = mk("buffer_view"); *out = as<VkBufferView>(h);
  W w = rec("vkCreateBufferView");
  w.k("view").h(h).k("buffer").h(H(ci->buffer)).k("format").en(name_VkFormat(ci->format), ci->format)
   .k("offset").u(ci->offset).k("range");
  if (ci->range == VK_WHOLE_SIZE) w.str("whole"); else w.u(ci->range);
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(VkDevice, const VkSamplerCreateInfo* ci, const VkAllocationCallbacks*, VkSampler* out) {
  ENTER;
  uint64_t h = mk("sampler"); *out = as<VkSampler>(h);
  W w = rec("vkCreateSampler");
  w.k("sampler").h(h).k("mag_filter").en(name_VkFilter(ci->magFilter), ci->magFilter)
   .k("min_filter").en(name_VkFilter(ci->minFilter), ci->minFilter)
   .k("mipmap_mode").en(name_VkSamplerMipmapMode(ci->mipmapMode), ci->mipmapMode)
   .k("address_u").en(name_VkSamplerAddressMode(ci->addressModeU), ci->addressModeU)
   .k("address_v").en(name_VkSamplerAddressMode(ci->addressModeV), ci->addressModeV)
   .k("address_w").en(name_VkSamplerAddressMode(ci->addressModeW), ci->addressModeW)
   .k("compare").b(ci->compareEnable).k("anisotropy").b(ci->anisotropyEnable);
  emit(w);
  return VK_SUCCESS;
}

// ---------------------------------------------------------------- shaders, layouts, pipelines
VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks*,
                                                    VkShaderModule* out) {
  ENTER;
  uint64_t h = mk("shader_module"); *out = as<VkShaderModule>(h);
  Sha1 s; s.update(ci->pCode, ci->codeSize);
  std::string sha = s.hex();
  g_module_sha[h] = sha;
  if (const char* dir = getenv("COVENANT_SPIRV_DIR")) {
    std::string p = std::string(dir) + "/" + sha + ".spv";
    if (FILE* f = fopen(p.c_str(), "wb")) { fwrite(ci->pCode, 1, ci->codeSize, f); fclose(f); }
  }
  W w = rec("vkCreateShaderModule");
  w.k("module").h(h).k("sha1").str(sha).k("code_size").u(ci->codeSize);
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice, const VkDescriptorSetLayoutCreateInfo* ci,
                                                           const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
  ENTER;
  uint64_t h = mk("descriptor_set_layout"); *out = as<VkDescriptorSetLayout>(h);
  auto* bf = find_chain<VkDescriptorSetLayoutBindingFlagsCreateInfo>(ci->pNext,
             VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
  W w = rec("vkCreateDescriptorSetLayout");
  w.k("layout").h(h).k("create_flags").u(ci->flags).k("bindings").ab();
  for (uint32_t i = 0; i < ci->bindingCount; i++) {
    auto& b = ci->pBindings[i];
    w.ob().k("binding").u(b.binding).k("type").en(name_VkDescriptorType(b.descriptorType), b.descriptorType)
     .k("count").u(b.descriptorCount).k("stages").fl(flags_VkShaderStageFlagBits, b.stageFlags)
     .k("immutable_samplers").b(b.pImmutableSamplers != nullptr);
    if (bf && i < bf->bindingCount) w.k("flags").u(bf->pBindingFlags[i]);
    w.oe();
  }
  w.ae(); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo* ci, const VkAllocationCallbacks*,
                                                      VkPipelineLayout* out) {
  ENTER;
  uint64_t h = mk("pipeline_layout"); *out = as<VkPipelineLayout>(h);
  W w = rec("vkCreatePipelineLayout");
  w.k("layout").h(h).k("set_layouts").ab();
  for (uint32_t i = 0; i < ci->setLayoutCount; i++) w.h(H(ci->pSetLayouts[i]));
  w.ae().k("push_constants").ab();
  for (uint32_t i = 0; i < ci->pushConstantRangeCount; i++) {
    auto& r = ci->pPushConstantRanges[i];
    w.ob().k("stages").fl(flags_VkShaderStageFlagBits, r.stageFlags).k("offset").u(r.offset).k("size").u(r.size).oe();
  }
  w.ae(); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo* ci, const VkAllocationCallbacks*,
                                                      VkDescriptorPool* out) {
  ENTER;
  uint64_t h = mk("descriptor_pool"); *out = as<VkDescriptorPool>(h);
  W w = rec("vkCreateDescriptorPool");
  w.k("pool").h(h).k("max_sets").u(ci->maxSets).k("sizes").ab();
  for (uint32_t i = 0; i < ci->poolSizeCount; i++)
    w.ob().k("type").en(name_VkDescriptorType(ci->pPoolSizes[i].type), ci->pPoolSizes[i].type)
     .k("count").u(ci->pPoolSizes[i].descriptorCount).oe();
  w.ae(); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo* ai, VkDescriptorSet* out) {
  ENTER;
  W w = rec("vkAllocateDescriptorSets");
  w.k("pool").h(H(ai->descriptorPool)).k("sets").ab();
  for (uint32_t i = 0; i < ai->descriptorSetCount; i++) { uint64_t h = mk("descriptor_set"); out[i] = as<VkDescriptorSet>(h); w.h(h); }
  w.ae().k("set_layouts").ab();
  for (uint32_t i = 0; i < ai->descriptorSetCount; i++) w.h(H(ai->pSetLayouts[i]));
  w.ae(); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkFreeDescriptorSets(VkDevice, VkDescriptorPool p, uint32_t n, const VkDescriptorSet* s) {
  ENTER; W w = rec("vkFreeDescriptorSets"); w.k("pool").h(H(p)).k("sets").ab();
  for (uint32_t i = 0; i < n; i++) w.h(H(s[i]));
  w.ae(); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice, VkDescriptorPool p, VkDescriptorPoolResetFlags) {
  ENTER; W w = rec("vkResetDescriptorPool"); w.k("pool").h(H(p)); emit(w); return VK_SUCCESS;
}
static void w_write(W& w, const VkWriteDescriptorSet& d) {
  w.ob().k("set").h(H(d.dstSet)).k("binding").u(d.dstBinding).k("array_element").u(d.dstArrayElement)
   .k("type").en(name_VkDescriptorType(d.descriptorType), d.descriptorType).k("elements").ab();
  for (uint32_t i = 0; i < d.descriptorCount; i++) {
    switch (d.descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER: case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
        auto& ii = d.pImageInfo[i];
        w.ob().k("sampler").h(H(ii.sampler)).k("view").h(H(ii.imageView))
         .k("layout").en(name_VkImageLayout(ii.imageLayout), ii.imageLayout).oe();
        break;
      }
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        w.ob().k("texel_view").h(H(d.pTexelBufferView[i])).oe(); break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: {
        auto& bi = d.pBufferInfo[i];
        w.ob().k("buffer").h(H(bi.buffer)).k("offset").u(bi.offset).k("range");
        if (bi.range == VK_WHOLE_SIZE) w.str("whole"); else w.u(bi.range);
        w.oe();
        break;
      }
      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
        auto* a = find_chain<VkWriteDescriptorSetAccelerationStructureKHR>(
            d.pNext, VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR);
        w.ob();
        if (a && i < a->accelerationStructureCount) w.k("acceleration_structure").h(H(a->pAccelerationStructures[i]));
        w.oe();
        break;
      }
      default: w.ob().oe(); break;
    }
  }
  w.ae().oe();
}
VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice, uint32_t nw, const VkWriteDescriptorSet* wr, uint32_t nc,
                                                  const VkCopyDescriptorSet* cp) {
  ENTER;
  W w = rec("vkUpdateDescriptorSets");
  w.k("writes").ab();
  for (uint32_t i = 0; i < nw; i++) w_write(w, wr[i]);
  w.ae().k("copies").ab();
  for (uint32_t i = 0; i < nc; i++)
    w.ob().k("src_set").h(H(cp[i].srcSet)).k("src_binding").u(cp[i].srcBinding).k("src_array_element").u(cp[i].srcArrayElement)
     .k("dst_set").h(H(cp[i].dstSet)).k("dst_binding").u(cp[i].dstBinding).k("dst_array_element").u(cp[i].dstArrayElement)
     .k("count").u(cp[i].descriptorCount).oe();
  w.ae(); emit(w);
}

static void w_attach_ref(W& w, const VkAttachmentReference& r) {
  w.ob().k("attachment").i(r.attachment == VK_ATTACHMENT_UNUSED ? -1 : (int64_t)r.attachment)
   .k("layout").en(name_VkImageLayout(r.layout), r.layout).oe();
}
static void w_attach_ref2(W& w, const VkAttachmentReference2& r) {
  w.ob().k("attachment").i(r.attachment == VK_ATTACHMENT_UNUSED ? -1 : (int64_t)r.attachment)
   .k("layout").en(name_VkImageLayout(r.layout), r.layout).oe();
}
extern "C++" { template <class AD>
static void w_attach_desc(W& w, const AD& a) {
  w.ob().k("format").en(name_VkFormat(a.format), a.format).k("samples").u(a.samples)
   .k("load").en(name_VkAttachmentLoadOp(a.loadOp), a.loadOp).k("store").en(name_VkAttachmentStoreOp(a.storeOp), a.storeOp)
   .k("stencil_load").en(name_VkAttachmentLoadOp(a.stencilLoadOp), a.stencilLoadOp)
   .k("stencil_store").en(name_VkAttachmentStoreOp(a.stencilStoreOp), a.stencilStoreOp)
   .k("initial_layout").en(name_VkImageLayout(a.initialLayout), a.initialLayout)
   .k("final_layout").en(name_VkImageLayout(a.finalLayout), a.finalLayout).oe();
}
}  // extern "C++"
extern "C++" { template <class DEP>
static void w_dep(W& w, const DEP& d, int32_t view_offset = 0, const VkMemoryBarrier2* mb2 = nullptr) {
  w.ob().k("src_subpass").i(d.srcSubpass == VK_SUBPASS_EXTERNAL ? -1 : (int64_t)d.srcSubpass)
   .k("dst_subpass").i(d.dstSubpass == VK_SUBPASS_EXTERNAL ? -1 : (int64_t)d.dstSubpass);
  if (mb2) {
    // a VkMemoryBarrier2 chained to VkSubpassDependency2 replaces its four masks
    w.k("src_stage"); stages(w, mb2->srcStageMask); w.k("dst_stage"); stages(w, mb2->dstStageMask);
    w.k("src_access"); access(w, mb2->srcAccessMask); w.k("dst_access"); access(w, mb2->dstAccessMask);
  } else {
    w.k("src_stage"); stages(w, d.srcStageMask); w.k("dst_stage"); stages(w, d.dstStageMask);
    w.k("src_access"); access(w, d.srcAccessMask); w.k("dst_access"); access(w, d.dstAccessMask);
  }
  // BY_REGION / VIEW_LOCAL make the dependency framebuffer-local
  w.k("dependency_flags").fl(flags_VkDependencyFlagBits, d.dependencyFlags);
  w.k("view_offset").i(view_offset);
  w.oe();
}
}  // extern "C++"
VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass(VkDevice, const VkRenderPassCreateInfo* ci, const VkAllocationCallbacks*,
                                                  VkRenderPass* out) {
  ENTER;
  uint64_t h = mk("render_pass"); *out = as<VkRenderPass>(h);
  W w = rec("vkCreateRenderPass");
  w.k("render_pass").h(h).k("attachments").ab();
  for (uint32_t i = 0; i < ci->attachmentCount; i++) w_attach_desc(w, ci->pAttachments[i]);
  w.ae().k("subpasses").ab();
  for (uint32_t i = 0; i < ci->subpassCount; i++) {
    auto& s = ci->pSubpasses[i];
    w.ob().k("color").ab(); for (uint32_t j = 0; j < s.colorAttachmentCount; j++) w_attach_ref(w, s.pColorAttachments[j]); w.ae();
    w.k("input").ab(); for (uint32_t j = 0; j < s.inputAttachmentCount; j++) w_attach_ref(w, s.pInputAttachments[j]); w.ae();
    w.k("resolve").ab(); if (s.pResolveAttachments) for (uint32_t j = 0; j < s.colorAttachmentCount; j++) w_attach_ref(w, s.pResolveAttachments[j]); w.ae();
    w.k("depth"); if (s.pDepthStencilAttachment) w_attach_ref(w, *s.pDepthStencilAttachment); else w.nul();
    w.oe();
  }
  auto* mv = find_chain<VkRenderPassMultiviewCreateInfo>(ci->pNext, VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO);
  w.ae().k("dependencies").ab();
  for (uint32_t i = 0; i < ci->dependencyCount; i++)
    w_dep(w, ci->pDependencies[i], (mv && i < mv->dependencyCount && mv->pViewOffsets) ? mv->pViewOffsets[i] : 0);
  w.ae().k("view_masks").ab();      // multiview (empty: none)
  if (mv && mv->subpassCount) for (uint32_t i = 0; i < mv->subpassCount; i++) w.u(mv->pViewMasks[i]);
  w.ae(); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass2(VkDevice, const VkRenderPassCreateInfo2* ci, const VkAllocationCallbacks*,
                                                   VkRenderPass* out) {
  ENTER;
  uint64_t h = mk("render_pass"); *out = as<VkRenderPass>(h);
  W w = rec("vkCreateRenderPass");     // same schema as vkCreateRenderPass
  w.k("render_pass").h(h).k("attachments").ab();
  for (uint32_t i = 0; i < ci->attachmentCount; i++) w_attach_desc(w, ci->pAttachments[i]);
  w.ae().k("subpasses").ab();
  for (uint32_t i = 0; i < ci->subpassCount; i++) {
    auto& s = ci->pSubpasses[i];
    w.ob().k("color").ab(); for (uint32_t j = 0; j < s.colorAttachmentCount; j++) w_attach_ref2(w, s.pColorAttachments[j]); w.ae();
    w.k("input").ab(); for (uint32_t j = 0; j < s.inputAttachmentCount; j++) w_attach_ref2(w, s.pInputAttachments[j]); w.ae();
    w.k("resolve").ab(); if (s.pResolveAttachments) for (uint32_t j = 0; j < s.colorAttachmentCount; j++) w_attach_ref2(w, s.pResolveAttachments[j]); w.ae();
    w.k("depth"); if (s.pDepthStencilAttachment) w_attach_ref2(w, *s.pDepthStencilAttachment); else w.nul();
    w.oe();
  }
  w.ae().k("dependencies").ab();
  for (uint32_t i = 0; i < ci->dependencyCount; i++) {
    auto& d = ci->pDependencies[i];
    w_dep(w, d, d.viewOffset, find_chain<VkMemoryBarrier2>(d.pNext, VK_STRUCTURE_TYPE_MEMORY_BARRIER_2));
  }
  w.ae().k("view_masks").ab();      // multiview (all zero: none)
  bool anyv = false;
  for (uint32_t i = 0; i < ci->subpassCount; i++) anyv |= ci->pSubpasses[i].viewMask != 0;
  if (anyv) for (uint32_t i = 0; i < ci->subpassCount; i++) w.u(ci->pSubpasses[i].viewMask);
  w.ae(); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateFramebuffer(VkDevice, const VkFramebufferCreateInfo* ci, const VkAllocationCallbacks*,
                                                   VkFramebuffer* out) {
  ENTER;
  uint64_t h = mk("framebuffer"); *out = as<VkFramebuffer>(h);
  W w = rec("vkCreateFramebuffer");
  w.k("framebuffer").h(h).k("render_pass").h(H(ci->renderPass)).k("attachments").ab();
  if (!(ci->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT))
    for (uint32_t i = 0; i < ci->attachmentCount; i++) w.h(H(ci->pAttachments[i]));
  w.ae().k("extent").ab().u(ci->width).u(ci->height).u(ci->layers).ae();
  emit(w);
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(VkDevice, VkPipelineCache, uint32_t n, const VkGraphicsPipelineCreateInfo* ci,
                                                         const VkAllocationCallbacks*, VkPipeline* out) {
  ENTER;
  W w = rec("vkCreateGraphicsPipelines");
  w.k("pipelines").ab();
  for (uint32_t i = 0; i < n; i++) {
    const auto& p = ci[i];
    uint64_t h = mk("pipeline"); out[i] = as<VkPipeline>(h);
    w.ob().k("pipeline").h(h).k("layout").h(H(p.layout)).k("render_pass").h(H(p.renderPass)).k("subpass").u(p.subpass);
    w_stages(w, p.stageCount, p.pStages);
    auto* r = find_chain<VkPipelineRenderingCreateInfo>(p.pNext, VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO);
    w.k("color_formats").ab();
    if (r && !p.renderPass) for (uint32_t j = 0; j < r->colorAttachmentCount; j++)
      w.en(name_VkFormat(r->pColorAttachmentFormats[j]), r->pColorAttachmentFormats[j]);
    w.ae();
    w.k("depth_format"); if (r && !p.renderPass) w.en(name_VkFormat(r->depthAttachmentFormat), r->depthAttachmentFormat); else w.nul();
    w.k("stencil_format"); if (r && !p.renderPass) w.en(name_VkFormat(r->stencilAttachmentFormat), r->stencilAttachmentFormat); else w.nul();
    if (r && !p.renderPass) w.k("view_mask").u(r->viewMask);
    if (auto* vi = p.pVertexInputState) {
      w.k("vertex_bindings").ab();
      for (uint32_t j = 0; j < vi->vertexBindingDescriptionCount; j++) {
        auto& b = vi->pVertexBindingDescriptions[j];
        w.ob().k("binding").u(b.binding).k("stride").u(b.stride)
         .k("rate").en(name_VkVertexInputRate(b.inputRate), b.inputRate).oe();
      }
      w.ae().k("vertex_attributes").ab();
      for (uint32_t j = 0; j < vi->vertexAttributeDescriptionCount; j++) {
        auto& a = vi->pVertexAttributeDescriptions[j];
        w.ob().k("location").u(a.location).k("binding").u(a.binding)
         .k("format").en(name_VkFormat(a.format), a.format).k("offset").u(a.offset).oe();
      }
      w.ae();
    }
    if (p.pInputAssemblyState) w.k("topology").en(name_VkPrimitiveTopology(p.pInputAssemblyState->topology), p.pInputAssemblyState->topology);
    if (p.pRasterizationState)
      w.k("polygon_mode").en(name_VkPolygonMode(p.pRasterizationState->polygonMode), p.pRasterizationState->polygonMode)
       .k("cull_mode").u(p.pRasterizationState->cullMode)
       .k("front_face_ccw").b(p.pRasterizationState->frontFace == VK_FRONT_FACE_COUNTER_CLOCKWISE);
    if (p.pMultisampleState) w.k("samples").u(p.pMultisampleState->rasterizationSamples);
    if (auto* ds = p.pDepthStencilState)
      w.k("depth_test").b(ds->depthTestEnable).k("depth_write").b(ds->depthWriteEnable)
       .k("depth_compare").en(name_VkCompareOp(ds->depthCompareOp), ds->depthCompareOp).k("stencil_test").b(ds->stencilTestEnable);
    if (auto* cb = p.pColorBlendState) {
      w.k("blend").ab();
      for (uint32_t j = 0; j < cb->attachmentCount; j++) {
        const auto& ba = cb->pAttachments[j];
        w.ob().k("enable").b(ba.blendEnable).k("write_mask").u(ba.colorWriteMask)
         .k("src_color").en(name_VkBlendFactor(ba.srcColorBlendFactor), ba.srcColorBlendFactor)
         .k("dst_color").en(name_VkBlendFactor(ba.dstColorBlendFactor), ba.dstColorBlendFactor)
         .k("color_op").en(name_VkBlendOp(ba.colorBlendOp), ba.colorBlendOp)
         .k("src_alpha").en(name_VkBlendFactor(ba.srcAlphaBlendFactor), ba.srcAlphaBlendFactor)
         .k("dst_alpha").en(name_VkBlendFactor(ba.dstAlphaBlendFactor), ba.dstAlphaBlendFactor)
         .k("alpha_op").en(name_VkBlendOp(ba.alphaBlendOp), ba.alphaBlendOp).oe();
      }
      w.ae();
    }
    // static viewports (ignored by Vulkan, and possibly dangling, when the viewport is dynamic)
    bool dyn_vp = false;
    if (auto* dy = p.pDynamicState)
      for (uint32_t j = 0; j < dy->dynamicStateCount; j++)
        dyn_vp |= dy->pDynamicStates[j] == VK_DYNAMIC_STATE_VIEWPORT ||
                  dy->pDynamicStates[j] == VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT;
    if (auto* vs = p.pViewportState; vs && !dyn_vp && vs->pViewports) {
      w.k("viewports").ab();
      for (uint32_t j = 0; j < vs->viewportCount; j++) {
        const auto& v = vs->pViewports[j];
        w.ab().d(v.x).d(v.y).d(v.width).d(v.height).d(v.minDepth).d(v.maxDepth).ae();
      }
      w.ae();
    }
    if (auto* dy = p.pDynamicState) {
      w.k("dynamic_states").ab();
      for (uint32_t j = 0; j < dy->dynamicStateCount; j++) w.en(name_VkDynamicState(dy->pDynamicStates[j]), dy->pDynamicStates[j]);
      w.ae();
    }
    w.oe();
  }
  w.ae(); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice, VkPipelineCache, uint32_t n, const VkComputePipelineCreateInfo* ci,
                                                        const VkAllocationCallbacks*, VkPipeline* out) {
  ENTER;
  W w = rec("vkCreateComputePipelines");
  w.k("pipelines").ab();
  for (uint32_t i = 0; i < n; i++) {
    uint64_t h = mk("pipeline"); out[i] = as<VkPipeline>(h);
    w.ob().k("pipeline").h(h).k("layout").h(H(ci[i].layout));
    w_stages(w, 1, &ci[i].stage);
    w.oe();
  }
  w.ae(); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineCache(VkDevice, const VkPipelineCacheCreateInfo*, const VkAllocationCallbacks*, VkPipelineCache* out) {
  ENTER; uint64_t h = mk("pipeline_cache"); *out = as<VkPipelineCache>(h);
  W w = rec("vkCreatePipelineCache"); w.k("cache").h(h); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetPipelineCacheData(VkDevice, VkPipelineCache, size_t* size, void*) { *size = 0; return VK_SUCCESS; }

// ---------------------------------------------------------------- command pools / buffers
VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice, const VkCommandPoolCreateInfo* ci, const VkAllocationCallbacks*,
                                                   VkCommandPool* out) {
  ENTER;
  uint64_t h = mk("command_pool"); *out = as<VkCommandPool>(h);
  W w = rec("vkCreateCommandPool");
  w.k("pool").h(h).k("family").u(ci->queueFamilyIndex).k("flags").fl(flags_VkCommandPoolCreateFlagBits, ci->flags);
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandPool(VkDevice, VkCommandPool p, VkCommandPoolResetFlags) {
  ENTER; W w = rec("vkResetCommandPool"); w.k("pool").h(H(p)); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo* ai, VkCommandBuffer* out) {
  ENTER;
  W w = rec("vkAllocateCommandBuffers");
  w.k("pool").h(H(ai->commandPool)).k("level").en(name_VkCommandBufferLevel(ai->level), ai->level).k("command_buffers").ab();
  for (uint32_t i = 0; i < ai->commandBufferCount; i++) { uint64_t h = mk("command_buffer"); out[i] = as<VkCommandBuffer>(h); w.h(h); }
  w.ae(); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkFreeCommandBuffers(VkDevice, VkCommandPool p, uint32_t n, const VkCommandBuffer* cbs) {
  ENTER; W w = rec("vkFreeCommandBuffers"); w.k("pool").h(H(p)).k("command_buffers").ab();
  for (uint32_t i = 0; i < n; i++) w.h(H(cbs[i]));
  w.ae(); emit(w);
}
VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* bi) {
  ENTER;
  g_readops.erase(H(cb)); g_tlas_pending.erase(H(cb));
  W w = rec("vkBeginCommandBuffer");
  w.k("cb").h(H(cb)).k("flags").fl(flags_VkCommandBufferUsageFlagBits, bi->flags);
  if (bi->pInheritanceInfo) {
    auto* in = bi->pInheritanceInfo;
    w.k("inherit_render_pass").h(H(in->renderPass)).k("inherit_framebuffer").h(H(in->framebuffer));
    if (auto* r = find_chain<VkCommandBufferInheritanceRenderingInfo>(in->pNext, VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO)) {
      w.k("inherit_color_formats").ab();
      for (uint32_t j = 0; j < r->colorAttachmentCount; j++) w.en(name_VkFormat(r->pColorAttachmentFormats[j]), r->pColorAttachmentFormats[j]);
      w.ae();
    }
  }
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer cb) {
  ENTER; W w = rec("vkEndCommandBuffer"); w.k("cb").h(H(cb)); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer cb, VkCommandBufferResetFlags) {
  ENTER; g_readops.erase(H(cb)); g_tlas_pending.erase(H(cb));
  W w = rec("vkResetCommandBuffer"); w.k("cb").h(H(cb)); emit(w); return VK_SUCCESS;
}

// ---------------------------------------------------------------- commands
VKAPI_ATTR void VKAPI_CALL vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo* ri) {
  ENTER;
  W w = rec("vkCmdBeginRendering");
  w.k("cb").h(H(cb)).k("area").ab().i(ri->renderArea.offset.x).i(ri->renderArea.offset.y)
   .u(ri->renderArea.extent.width).u(ri->renderArea.extent.height).ae()
   .k("layer_count").u(ri->layerCount).k("view_mask").u(ri->viewMask).k("flags").u(ri->flags);
  w.k("color").ab();
  for (uint32_t i = 0; i < ri->colorAttachmentCount; i++) w_attachment(w, &ri->pColorAttachments[i]);
  w.ae();
  w.k("depth"); w_attachment(w, ri->pDepthAttachment);
  w.k("stencil"); w_attachment(w, ri->pStencilAttachment);
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderingKHR(VkCommandBuffer cb, const VkRenderingInfo* ri) { vkCmdBeginRendering(cb, ri); }
VKAPI_ATTR void VKAPI_CALL vkCmdEndRendering(VkCommandBuffer cb) {
  ENTER; W w = rec("vkCmdEndRendering"); w.k("cb").h(H(cb)); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderingKHR(VkCommandBuffer cb) { vkCmdEndRendering(cb); }
VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi, VkSubpassContents c) {
  ENTER;
  W w = rec("vkCmdBeginRenderPass");
  w.k("cb").h(H(cb)).k("render_pass").h(H(bi->renderPass)).k("framebuffer").h(H(bi->framebuffer))
   .k("area").ab().i(bi->renderArea.offset.x).i(bi->renderArea.offset.y)
   .u(bi->renderArea.extent.width).u(bi->renderArea.extent.height).ae()
   .k("contents").en(name_VkSubpassContents(c), c);
  w.k("clear_values").ab();
  for (uint32_t i = 0; i < bi->clearValueCount; i++) {
    const VkClearValue& cv = bi->pClearValues[i];
    w.ob().k("color").ab().d(cv.color.float32[0]).d(cv.color.float32[1]).d(cv.color.float32[2])
     .d(cv.color.float32[3]).ae().k("depth").d(cv.depthStencil.depth).k("stencil").u(cv.depthStencil.stencil).oe();
  }
  w.ae();
  if (auto* a = find_chain<VkRenderPassAttachmentBeginInfo>(bi->pNext, VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO)) {
    w.k("attachments").ab(); for (uint32_t i = 0; i < a->attachmentCount; i++) w.h(H(a->pAttachments[i])); w.ae();
  }
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass2(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi, const VkSubpassBeginInfo* si) {
  vkCmdBeginRenderPass(cb, bi, si->contents);
}
VKAPI_ATTR void VKAPI_CALL vkCmdNextSubpass(VkCommandBuffer cb, VkSubpassContents c) {
  ENTER; W w = rec("vkCmdNextSubpass"); w.k("cb").h(H(cb)).k("contents").en(name_VkSubpassContents(c), c); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdNextSubpass2(VkCommandBuffer cb, const VkSubpassBeginInfo* si, const VkSubpassEndInfo*) {
  vkCmdNextSubpass(cb, si->contents);
}
VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass(VkCommandBuffer cb) {
  ENTER; W w = rec("vkCmdEndRenderPass"); w.k("cb").h(H(cb)); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass2(VkCommandBuffer cb, const VkSubpassEndInfo*) { vkCmdEndRenderPass(cb); }

VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {
  ENTER; W w = rec("vkCmdBindPipeline");
  w.k("cb").h(H(cb)).k("bind_point").en(name_VkPipelineBindPoint(bp), bp).k("pipeline").h(H(p)); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first,
                                                   uint32_t n, const VkDescriptorSet* sets, uint32_t nd, const uint32_t* dyn) {
  ENTER; W w = rec("vkCmdBindDescriptorSets");
  w.k("cb").h(H(cb)).k("bind_point").en(name_VkPipelineBindPoint(bp), bp).k("layout").h(H(l)).k("first_set").u(first).k("sets").ab();
  for (uint32_t i = 0; i < n; i++) w.h(H(sets[i]));
  w.ae().k("dynamic_offsets").ab();
  for (uint32_t i = 0; i < nd; i++) w.u(dyn[i]);
  w.ae(); emit(w);
}
static void push_descriptor_set(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t set,
                                uint32_t n, const VkWriteDescriptorSet* wr) {
  W w = rec("vkCmdPushDescriptorSetKHR");
  w.k("cb").h(H(cb)).k("bind_point").en(name_VkPipelineBindPoint(bp), bp).k("layout").h(H(l)).k("set").u(set)
   .k("writes").ab();
  for (uint32_t i = 0; i < n; i++) w_write(w, wr[i]);   // dstSet is ignored (null)
  w.ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdPushDescriptorSetKHR(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l,
                                                     uint32_t set, uint32_t n, const VkWriteDescriptorSet* wr) {
  ENTER; push_descriptor_set(cb, bp, l, set, n, wr);
}
#ifdef VK_VERSION_1_4
VKAPI_ATTR void VKAPI_CALL vkCmdPushDescriptorSet(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l,
                                                  uint32_t set, uint32_t n, const VkWriteDescriptorSet* wr) {
  ENTER; push_descriptor_set(cb, bp, l, set, n, wr);
}
VKAPI_ATTR void VKAPI_CALL vkCmdPushDescriptorSet2(VkCommandBuffer cb, const VkPushDescriptorSetInfo* pi) {
  ENTER;
  // one record per bind point the stages reach
  if (pi->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
    push_descriptor_set(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pi->layout, pi->set, pi->descriptorWriteCount, pi->pDescriptorWrites);
  if (pi->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS)
    push_descriptor_set(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pi->layout, pi->set, pi->descriptorWriteCount, pi->pDescriptorWrites);
}
#endif
VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off,
                                              uint32_t size, const void* data) {
  ENTER; W w = rec("vkCmdPushConstants");
  w.k("cb").h(H(cb)).k("layout").h(H(l)).k("stages").fl(flags_VkShaderStageFlagBits, st).k("offset").u(off).k("size").u(size);
  w.k("data").hex(data, std::min<uint32_t>(size, 256));
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer* b, const VkDeviceSize* o) {
  ENTER; W w = rec("vkCmdBindVertexBuffers");
  w.k("cb").h(H(cb)).k("first_binding").u(first).k("buffers").ab();
  for (uint32_t i = 0; i < n; i++) w.h(H(b[i]));
  w.ae().k("offsets").ab();
  for (uint32_t i = 0; i < n; i++) w.u(o[i]);
  w.ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer* b,
                                                   const VkDeviceSize* o, const VkDeviceSize*, const VkDeviceSize*) {
  vkCmdBindVertexBuffers(cb, first, n, b, o);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t) {
  ENTER; W w = rec("vkCmdBindIndexBuffer");
  w.k("cb").h(H(cb)).k("buffer").h(H(b)).k("offset").u(o).k("index_type").en(name_VkIndexType(t), t); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) {
  ENTER; W w = rec("vkCmdDraw");
  w.k("cb").h(H(cb)).k("vertex_count").u(vc).k("instance_count").u(ic).k("first_vertex").u(fv).k("first_instance").u(fi); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo, uint32_t finst) {
  ENTER; W w = rec("vkCmdDrawIndexed");
  w.k("cb").h(H(cb)).k("index_count").u(ic).k("instance_count").u(inst).k("first_index").u(fi)
   .k("vertex_offset").i(vo).k("first_instance").u(finst); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t stride) {
  ENTER; W w = rec("vkCmdDrawIndirect");
  w.k("cb").h(H(cb)).k("buffer").h(H(b)).k("offset").u(o).k("draw_count").u(n).k("stride").u(stride); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t stride) {
  ENTER; W w = rec("vkCmdDrawIndexedIndirect");
  w.k("cb").h(H(cb)).k("buffer").h(H(b)).k("offset").u(o).k("draw_count").u(n).k("stride").u(stride); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
  ENTER; W w = rec("vkCmdDispatch");
  w.k("cb").h(H(cb)).k("groups").ab().u(x).u(y).u(z).ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o) {
  ENTER; W w = rec("vkCmdDispatchIndirect"); w.k("cb").h(H(cb)).k("buffer").h(H(b)).k("offset").u(o); emit(w);
}
// ---------------------------------------------------------------- compute / draw variants
VKAPI_ATTR void VKAPI_CALL vkCmdDispatchBase(VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz,
                                             uint32_t x, uint32_t y, uint32_t z) {
  ENTER; W w = rec("vkCmdDispatchBase");
  w.k("cb").h(H(cb)).k("base").ab().u(bx).u(by).u(bz).ae().k("groups").ab().u(x).u(y).u(z).ae(); emit(w);
}
static void indirect_count(const char* fn, VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cbuf,
                           VkDeviceSize co, uint32_t maxn, uint32_t stride) {
  W w = rec(fn);
  w.k("cb").h(H(cb)).k("buffer").h(H(b)).k("offset").u(o).k("count_buffer").h(H(cbuf)).k("count_offset").u(co)
   .k("max_draw_count").u(maxn).k("stride").u(stride);
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cbuf,
                                                  VkDeviceSize co, uint32_t maxn, uint32_t stride) {
  ENTER; indirect_count("vkCmdDrawIndirectCount", cb, b, o, cbuf, co, maxn, stride);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cbuf,
                                                         VkDeviceSize co, uint32_t maxn, uint32_t stride) {
  ENTER; indirect_count("vkCmdDrawIndexedIndirectCount", cb, b, o, cbuf, co, maxn, stride);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndirectCountAMD(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cbuf,
                                                     VkDeviceSize co, uint32_t maxn, uint32_t stride) {
  vkCmdDrawIndirectCount(cb, b, o, cbuf, co, maxn, stride);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirectCountAMD(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cbuf,
                                                            VkDeviceSize co, uint32_t maxn, uint32_t stride) {
  vkCmdDrawIndexedIndirectCount(cb, b, o, cbuf, co, maxn, stride);
}

// ---------------------------------------------------------------- mesh shading (VK_EXT_mesh_shader)
VKAPI_ATTR void VKAPI_CALL vkCmdDrawMeshTasksEXT(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
  ENTER; W w = rec("vkCmdDrawMeshTasksEXT");
  w.k("cb").h(H(cb)).k("groups").ab().u(x).u(y).u(z).ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawMeshTasksIndirectEXT(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t stride) {
  ENTER; W w = rec("vkCmdDrawMeshTasksIndirectEXT");
  w.k("cb").h(H(cb)).k("buffer").h(H(b)).k("offset").u(o).k("draw_count").u(n).k("stride").u(stride); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cbuf,
                                                              VkDeviceSize co, uint32_t maxn, uint32_t stride) {
  ENTER; indirect_count("vkCmdDrawMeshTasksIndirectCountEXT", cb, b, o, cbuf, co, maxn, stride);
}

// ---------------------------------------------------------------- acceleration structures (VK_KHR_acceleration_structure)
static const char* as_type_name(VkAccelerationStructureTypeKHR t) {
  return t == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR ? "top" :
         t == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR ? "bottom" : "generic";
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateAccelerationStructureKHR(VkDevice, const VkAccelerationStructureCreateInfoKHR* ci,
                                                                const VkAllocationCallbacks*, VkAccelerationStructureKHR* out) {
  ENTER;
  uint64_t h = mk("acceleration_structure"); *out = as<VkAccelerationStructureKHR>(h);
  AccelStruct& a = g_as[h];
  a.buffer = H(ci->buffer); a.offset = ci->offset; a.size = ci->size; a.type = ci->type;
  a.addr = buf_addr(a.buffer) + ci->offset;
  g_asaddr[a.addr] = h;
  W w = rec("vkCreateAccelerationStructureKHR");
  w.k("acceleration_structure").h(h).k("buffer").h(a.buffer).k("offset").u(a.offset).k("size").u(a.size)
   .k("type").str(as_type_name(ci->type)).k("address").u(a.addr);
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyAccelerationStructureKHR(VkDevice, VkAccelerationStructureKHR h, const VkAllocationCallbacks*) {
  ENTER; if (!h) return;
  auto it = g_as.find(H(h));
  if (it != g_as.end()) { g_asaddr.erase(it->second.addr); g_as.erase(it); }
  W w = rec("vkDestroyAccelerationStructureKHR"); w.k("handle").h(H(h)); emit(w);
}
VKAPI_ATTR VkDeviceAddress VKAPI_CALL vkGetAccelerationStructureDeviceAddressKHR(VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR* i) {
  ENTER;
  auto it = g_as.find(H(i->accelerationStructure));
  VkDeviceAddress a = it == g_as.end() ? 0 : it->second.addr;
  W w = rec("vkGetAccelerationStructureDeviceAddressKHR");
  w.k("acceleration_structure").h(H(i->accelerationStructure)).k("address").u(a); emit(w);
  return a;
}
static uint64_t build_prims(const VkAccelerationStructureBuildGeometryInfoKHR& b, const uint32_t* counts) {
  uint64_t n = 0;
  for (uint32_t g = 0; g < b.geometryCount; g++) n += counts ? counts[g] : 0;
  return n;
}
VKAPI_ATTR void VKAPI_CALL vkGetAccelerationStructureBuildSizesKHR(VkDevice, VkAccelerationStructureBuildTypeKHR,
    const VkAccelerationStructureBuildGeometryInfoKHR* b, const uint32_t* maxPrims, VkAccelerationStructureBuildSizesInfoKHR* s) {
  uint64_t n = build_prims(*b, maxPrims);
  s->accelerationStructureSize = as_size(n);
  s->buildScratchSize = as_scratch(n, false);
  s->updateScratchSize = as_scratch(n, true);
}
VKAPI_ATTR void VKAPI_CALL vkGetDeviceAccelerationStructureCompatibilityKHR(VkDevice, const VkAccelerationStructureVersionInfoKHR*,
                                                                           VkAccelerationStructureCompatibilityKHR* c) {
  *c = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR;
}
static const VkAccelerationStructureGeometryKHR& geom_at(const VkAccelerationStructureBuildGeometryInfoKHR& b, uint32_t g) {
  return b.pGeometries ? b.pGeometries[g] : *b.ppGeometries[g];
}
static uint32_t index_size(VkIndexType t) {
  return t == VK_INDEX_TYPE_UINT16 ? 2 : t == VK_INDEX_TYPE_UINT32 ? 4 :
         (t == VK_INDEX_TYPE_UINT8_EXT ? 1 : 0);
}
// one build of a vkCmdBuildAccelerationStructures*KHR call (device: true -> device addresses)
static void w_build(W& w, uint64_t cb, uint32_t bi, const VkAccelerationStructureBuildGeometryInfoKHR& b,
                    const VkAccelerationStructureBuildRangeInfoKHR* ranges, const uint32_t* maxCounts, bool device) {
  bool update = b.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
  w.ob().k("type").str(as_type_name(b.type)).k("mode").str(update ? "update" : "build").k("flags").u(b.flags)
   .k("src").h(update ? H(b.srcAccelerationStructure) : 0).k("dst").h(H(b.dstAccelerationStructure));
  uint64_t prims = 0;
  for (uint32_t g = 0; g < b.geometryCount; g++) prims += ranges ? ranges[g].primitiveCount : (maxCounts ? maxCounts[g] : 0);
  if (device) w_devrange(w, "scratch", b.scratchData.deviceAddress, as_scratch(prims, update));
  w.k("geometries").ab();
  for (uint32_t g = 0; g < b.geometryCount; g++) {
    const auto& G = geom_at(b, g);
    uint32_t pc = ranges ? ranges[g].primitiveCount : (maxCounts ? maxCounts[g] : 0);
    uint32_t po = ranges ? ranges[g].primitiveOffset : 0, fv = ranges ? ranges[g].firstVertex : 0;
    uint32_t to = ranges ? ranges[g].transformOffset : 0;
    w.ob().k("primitive_count").u(pc).k("known_count").b(ranges != nullptr);
    if (G.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
      const auto& t = G.geometry.triangles;
      uint32_t isz = index_size(t.indexType);
      w.k("type").str("triangles");
      if (device) {
        if (isz) {
          w_devrange(w, "vertex", t.vertexData.deviceAddress ? t.vertexData.deviceAddress + (VkDeviceSize)fv * t.vertexStride : 0,
                     (VkDeviceSize)(t.maxVertex + 1) * t.vertexStride);
          w_devrange(w, "index", t.indexData.deviceAddress ? t.indexData.deviceAddress + po : 0, (VkDeviceSize)3 * pc * isz);
        } else {
          w_devrange(w, "vertex", t.vertexData.deviceAddress ? t.vertexData.deviceAddress + po + (VkDeviceSize)fv * t.vertexStride : 0,
                     (VkDeviceSize)3 * pc * t.vertexStride);
        }
        w_devrange(w, "transform", t.transformData.deviceAddress ? t.transformData.deviceAddress + to : 0, 48);
      }
    } else if (G.geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
      w.k("type").str("aabbs");
      if (device) w_devrange(w, "data", G.geometry.aabbs.data.deviceAddress ? G.geometry.aabbs.data.deviceAddress + po : 0,
                             (VkDeviceSize)pc * G.geometry.aabbs.stride);
    } else if (G.geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
      const auto& in = G.geometry.instances;
      w.k("type").str("instances").k("array_of_pointers").b(in.arrayOfPointers);
      if (device && in.data.deviceAddress) {
        VkDeviceAddress a = in.data.deviceAddress + po;
        w_devrange(w, "data", a, (VkDeviceSize)pc * (in.arrayOfPointers ? 8 : 64));
        auto r = resolve_addr(a);
        if (r.first && ranges)
          g_tlas_pending[cb].push_back({g_seq, bi, r.first, r.second, pc, (bool)in.arrayOfPointers});
      }
    } else {
      w.k("type").u(G.geometryType);
    }
    w.oe();
  }
  w.ae().oe();
}
VKAPI_ATTR void VKAPI_CALL vkCmdBuildAccelerationStructuresKHR(VkCommandBuffer cb, uint32_t n,
    const VkAccelerationStructureBuildGeometryInfoKHR* infos, const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
  ENTER; W w = rec("vkCmdBuildAccelerationStructuresKHR");
  w.k("cb").h(H(cb)).k("builds").ab();
  for (uint32_t i = 0; i < n; i++) w_build(w, H(cb), i, infos[i], ranges[i], nullptr, true);
  w.ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBuildAccelerationStructuresIndirectKHR(VkCommandBuffer cb, uint32_t n,
    const VkAccelerationStructureBuildGeometryInfoKHR* infos, const VkDeviceAddress* addrs, const uint32_t* strides,
    const uint32_t* const* maxCounts) {
  ENTER; W w = rec("vkCmdBuildAccelerationStructuresIndirectKHR");
  w.k("cb").h(H(cb)).k("builds").ab();
  for (uint32_t i = 0; i < n; i++) w_build(w, H(cb), i, infos[i], nullptr, maxCounts[i], true);
  w.ae().k("indirect").ab();
  for (uint32_t i = 0; i < n; i++) {
    w.ob(); w_devrange(w, "range", addrs[i], (VkDeviceSize)std::max(1u, infos[i].geometryCount) * std::max<uint32_t>(strides[i], 16));
    w.oe();
  }
  w.ae(); emit(w);
}
static const char* copy_mode(VkCopyAccelerationStructureModeKHR m) {
  return m == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR ? "compact" :
         m == VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR ? "clone" :
         m == VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR ? "serialize" : "deserialize";
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyAccelerationStructureKHR(VkCommandBuffer cb, const VkCopyAccelerationStructureInfoKHR* i) {
  ENTER; W w = rec("vkCmdCopyAccelerationStructureKHR");
  w.k("cb").h(H(cb)).k("src").h(H(i->src)).k("dst").h(H(i->dst)).k("mode").str(copy_mode(i->mode)); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyAccelerationStructureToMemoryKHR(VkCommandBuffer cb, const VkCopyAccelerationStructureToMemoryInfoKHR* i) {
  ENTER; W w = rec("vkCmdCopyAccelerationStructureToMemoryKHR");
  auto it = g_as.find(H(i->src));
  w.k("cb").h(H(cb)).k("src").h(H(i->src)).k("mode").str(copy_mode(i->mode));
  w_devrange(w, "dst", i->dst.deviceAddress, (it == g_as.end() ? 0 : it->second.size) + 64);
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyMemoryToAccelerationStructureKHR(VkCommandBuffer cb, const VkCopyMemoryToAccelerationStructureInfoKHR* i) {
  ENTER; W w = rec("vkCmdCopyMemoryToAccelerationStructureKHR");
  auto it = g_as.find(H(i->dst));
  w.k("cb").h(H(cb)).k("dst").h(H(i->dst)).k("mode").str(copy_mode(i->mode));
  w_devrange(w, "src", i->src.deviceAddress, (it == g_as.end() ? 0 : it->second.size) + 64);
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdWriteAccelerationStructuresPropertiesKHR(VkCommandBuffer cb, uint32_t n, const VkAccelerationStructureKHR* a,
                                                                         VkQueryType t, VkQueryPool pool, uint32_t first) {
  ENTER; W w = rec("vkCmdWriteAccelerationStructuresPropertiesKHR");
  w.k("cb").h(H(cb)).k("acceleration_structures").ab();
  for (uint32_t i = 0; i < n; i++) w.h(H(a[i]));
  w.ae().k("query_type").u(t).k("query_pool").h(H(pool)).k("first_query").u(first); emit(w);
}
// host commands (accelerationStructureHostCommands): the host builds/copies in place
static VkResult host_done(VkDeferredOperationKHR op) { return op ? VK_OPERATION_NOT_DEFERRED_KHR : VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkBuildAccelerationStructuresKHR(VkDevice, VkDeferredOperationKHR op, uint32_t n,
    const VkAccelerationStructureBuildGeometryInfoKHR* infos, const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
  ENTER; W w = rec("vkBuildAccelerationStructuresKHR");
  w.k("deferred").b(op != VK_NULL_HANDLE).k("builds").ab();
  for (uint32_t i = 0; i < n; i++) w_build(w, 0, i, infos[i], ranges[i], nullptr, false);
  w.ae(); emit(w); return host_done(op);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCopyAccelerationStructureKHR(VkDevice, VkDeferredOperationKHR op, const VkCopyAccelerationStructureInfoKHR* i) {
  ENTER; W w = rec("vkCopyAccelerationStructureKHR");
  w.k("src").h(H(i->src)).k("dst").h(H(i->dst)).k("mode").str(copy_mode(i->mode)); emit(w); return host_done(op);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCopyAccelerationStructureToMemoryKHR(VkDevice, VkDeferredOperationKHR op, const VkCopyAccelerationStructureToMemoryInfoKHR* i) {
  ENTER; W w = rec("vkCopyAccelerationStructureToMemoryKHR"); w.k("src").h(H(i->src)); emit(w); return host_done(op);
}
VKAPI_ATTR VkResult VKAPI_CALL vkCopyMemoryToAccelerationStructureKHR(VkDevice, VkDeferredOperationKHR op, const VkCopyMemoryToAccelerationStructureInfoKHR* i) {
  ENTER; W w = rec("vkCopyMemoryToAccelerationStructureKHR"); w.k("dst").h(H(i->dst)); emit(w); return host_done(op);
}
VKAPI_ATTR VkResult VKAPI_CALL vkWriteAccelerationStructuresPropertiesKHR(VkDevice, uint32_t n, const VkAccelerationStructureKHR* a,
                                                                          VkQueryType, size_t size, void* data, size_t stride) {
  ENTER;
  W w = rec("vkWriteAccelerationStructuresPropertiesKHR"); w.k("acceleration_structures").ab();
  for (uint32_t i = 0; i < n; i++) {
    w.h(H(a[i]));
    auto it = g_as.find(H(a[i]));
    if (data && (i + 1) * stride <= size && stride >= 8) { uint64_t v = it == g_as.end() ? 4096 : it->second.size; memcpy((char*)data + i * stride, &v, 8); }
  }
  w.ae(); emit(w); return VK_SUCCESS;
}

// ---------------------------------------------------------------- queries (results only: no memory model)
VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(VkDevice, const VkQueryPoolCreateInfo* ci, const VkAllocationCallbacks*, VkQueryPool* out) {
  ENTER; uint64_t h = mk("query_pool"); *out = as<VkQueryPool>(h);
  g_query_type[h] = ci->queryType;
  W w = rec("vkCreateQueryPool"); w.k("pool").h(h).k("query_type").u(ci->queryType).k("count").u(ci->queryCount); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(VkDevice, VkQueryPool pool, uint32_t first, uint32_t n, size_t size, void* data,
                                                     VkDeviceSize stride, VkQueryResultFlags flags) {
  ENTER;
  // compacted / serialization sizes: a plausible size; other queries: 0 (timestamps, occlusion ...)
  uint32_t t = g_query_type.count(H(pool)) ? g_query_type[H(pool)] : 0;
  bool size_query = t == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR ||
                    t == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR ||
                    t == VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR;
  bool b64 = flags & VK_QUERY_RESULT_64_BIT, avail = flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
  for (uint32_t i = 0; i < n; i++) {
    char* p = (char*)data + i * stride;
    if ((i * stride) + (b64 ? 8 : 4) * (avail ? 2 : 1) > size) break;
    uint64_t v = size_query ? 1u << 20 : 0;
    if (b64) { memcpy(p, &v, 8); if (avail) { uint64_t one = 1; memcpy(p + 8, &one, 8); } }
    else { uint32_t v32 = (uint32_t)v; memcpy(p, &v32, 4); if (avail) { uint32_t one = 1; memcpy(p + 4, &one, 4); } }
  }
  W w = rec("vkGetQueryPoolResults");
  w.k("pool").h(H(pool)).k("first").u(first).k("count").u(n).k("wait").b(flags & VK_QUERY_RESULT_WAIT_BIT); emit(w);
  return VK_SUCCESS;
}

// ---------------------------------------------------------------- ray tracing pipelines (VK_KHR_ray_tracing_pipeline)
VKAPI_ATTR VkResult VKAPI_CALL vkCreateRayTracingPipelinesKHR(VkDevice, VkDeferredOperationKHR op, VkPipelineCache, uint32_t n,
    const VkRayTracingPipelineCreateInfoKHR* ci, const VkAllocationCallbacks*, VkPipeline* out) {
  ENTER;
  W w = rec("vkCreateRayTracingPipelinesKHR");
  w.k("pipelines").ab();
  for (uint32_t i = 0; i < n; i++) {
    const auto& p = ci[i];
    uint64_t h = mk("pipeline"); out[i] = as<VkPipeline>(h);
    w.ob().k("pipeline").h(h).k("layout").h(H(p.layout)).k("flags").u(p.flags)
     .k("max_recursion").u(p.maxPipelineRayRecursionDepth);
    w_stages(w, p.stageCount, p.pStages);
    w.k("groups").ab();
    for (uint32_t g = 0; g < p.groupCount; g++) {
      const auto& G = p.pGroups[g];
      auto idx = [&](uint32_t v) -> int64_t { return v == VK_SHADER_UNUSED_KHR ? -1 : (int64_t)v; };
      w.ob().k("type").u(G.type).k("general").i(idx(G.generalShader)).k("closest_hit").i(idx(G.closestHitShader))
       .k("any_hit").i(idx(G.anyHitShader)).k("intersection").i(idx(G.intersectionShader)).oe();
    }
    w.ae().k("libraries").ab();
    if (p.pLibraryInfo) for (uint32_t l = 0; l < p.pLibraryInfo->libraryCount; l++) w.h(H(p.pLibraryInfo->pLibraries[l]));
    w.ae();
    if (auto* dy = p.pDynamicState) {
      w.k("dynamic_states").ab();
      for (uint32_t j = 0; j < dy->dynamicStateCount; j++) w.en(name_VkDynamicState(dy->pDynamicStates[j]), dy->pDynamicStates[j]);
      w.ae();
    }
    w.oe();
  }
  w.ae(); emit(w);
  return host_done(op);
}
static void fill_group_handles(VkPipeline p, uint32_t first, uint32_t count, size_t size, void* data) {
  // 32 bytes per group: pipeline handle, group index, a tag
  for (uint32_t g = 0; g < count && (g + 1) * 32 <= size; g++) {
    uint64_t v[4] = {H(p), (uint64_t)(first + g), 0xC0DEC0DEull, 0};
    memcpy((char*)data + g * 32, v, 32);
  }
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetRayTracingShaderGroupHandlesKHR(VkDevice, VkPipeline p, uint32_t first, uint32_t count,
                                                                    size_t size, void* data) {
  ENTER; fill_group_handles(p, first, count, size, data);
  W w = rec("vkGetRayTracingShaderGroupHandlesKHR"); w.k("pipeline").h(H(p)).k("first").u(first).k("count").u(count); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(VkDevice, VkPipeline p, uint32_t first, uint32_t count,
                                                                                 size_t size, void* data) {
  ENTER; fill_group_handles(p, first, count, size, data); return VK_SUCCESS;
}
VKAPI_ATTR VkDeviceSize VKAPI_CALL vkGetRayTracingShaderGroupStackSizeKHR(VkDevice, VkPipeline, uint32_t, VkShaderGroupShaderKHR) {
  return 256;
}
static void w_sbt(W& w, const char* key, const VkStridedDeviceAddressRegionKHR* r) {
  if (!r || !r->deviceAddress || !r->size) { w.k(key).nul(); return; }
  w_devrange(w, key, r->deviceAddress, r->size);
}
static void trace_rays(const char* fn, VkCommandBuffer cb, const VkStridedDeviceAddressRegionKHR* rg,
                       const VkStridedDeviceAddressRegionKHR* ms, const VkStridedDeviceAddressRegionKHR* ht,
                       const VkStridedDeviceAddressRegionKHR* cl, VkDeviceAddress ind, VkDeviceSize ind_size,
                       uint32_t x, uint32_t y, uint32_t z) {
  W w = rec(fn);
  w.k("cb").h(H(cb));
  w_sbt(w, "raygen", rg); w_sbt(w, "miss", ms); w_sbt(w, "hit", ht); w_sbt(w, "callable", cl);
  if (ind) w_devrange(w, "indirect", ind, ind_size);
  else w.k("size").ab().u(x).u(y).u(z).ae();
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdTraceRaysKHR(VkCommandBuffer cb, const VkStridedDeviceAddressRegionKHR* rg,
    const VkStridedDeviceAddressRegionKHR* ms, const VkStridedDeviceAddressRegionKHR* ht, const VkStridedDeviceAddressRegionKHR* cl,
    uint32_t x, uint32_t y, uint32_t z) {
  ENTER; trace_rays("vkCmdTraceRaysKHR", cb, rg, ms, ht, cl, 0, 0, x, y, z);
}
VKAPI_ATTR void VKAPI_CALL vkCmdTraceRaysIndirectKHR(VkCommandBuffer cb, const VkStridedDeviceAddressRegionKHR* rg,
    const VkStridedDeviceAddressRegionKHR* ms, const VkStridedDeviceAddressRegionKHR* ht, const VkStridedDeviceAddressRegionKHR* cl,
    VkDeviceAddress ind) {
  ENTER; trace_rays("vkCmdTraceRaysIndirectKHR", cb, rg, ms, ht, cl, ind, sizeof(VkTraceRaysIndirectCommandKHR), 0, 0, 0);
}
VKAPI_ATTR void VKAPI_CALL vkCmdTraceRaysIndirect2KHR(VkCommandBuffer cb, VkDeviceAddress ind) {
  // the shader binding table addresses are device data: recorded as unknown
  ENTER; W w = rec("vkCmdTraceRaysIndirect2KHR");
  w.k("cb").h(H(cb)); w_devrange(w, "indirect", ind, sizeof(VkTraceRaysIndirectCommand2KHR));
  w.k("sbt_unknown").b(true); emit(w);
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport* v) {
  ENTER; W w = rec("vkCmdSetViewport"); w.k("cb").h(H(cb)).k("viewports").ab();
  for (uint32_t i = 0; i < n; i++) w.ab().d(v[i].x).d(v[i].y).d(v[i].width).d(v[i].height).d(v[i].minDepth).d(v[i].maxDepth).ae();
  w.ae(); (void)first; emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetViewportWithCount(VkCommandBuffer cb, uint32_t n, const VkViewport* v) {
  ENTER; W w = rec("vkCmdSetViewport"); w.k("cb").h(H(cb)).k("viewports").ab();
  for (uint32_t i = 0; i < n; i++) w.ab().d(v[i].x).d(v[i].y).d(v[i].width).d(v[i].height).d(v[i].minDepth).d(v[i].maxDepth).ae();
  w.ae().k("with_count").b(true); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetViewportWithCountEXT(VkCommandBuffer cb, uint32_t n, const VkViewport* v) {
  vkCmdSetViewportWithCount(cb, n, v);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D* r) {
  ENTER; W w = rec("vkCmdSetScissor"); w.k("cb").h(H(cb)).k("scissors").ab();
  for (uint32_t i = 0; i < n; i++) w.ab().i(r[i].offset.x).i(r[i].offset.y).u(r[i].extent.width).u(r[i].extent.height).ae();
  w.ae(); (void)first; emit(w);
}

static void w_buffer_copy(W& w, const char* fn, VkCommandBuffer cb, uint64_t src, uint64_t dst, uint32_t n,
                          const VkDeviceSize* so, const VkDeviceSize* dof, const VkDeviceSize* sz, size_t stride) {
  w.k("cb").h(H(cb)).k("src").h(src).k("dst").h(dst).k("regions").ab();
  for (uint32_t i = 0; i < n; i++) {
    auto at = [&](const VkDeviceSize* p) { return *(const VkDeviceSize*)((const char*)p + i * stride); };
    w.ob().k("src_offset").u(at(so)).k("dst_offset").u(at(dof)).k("size").u(at(sz)).oe();
  }
  w.ae(); (void)fn;
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy* r) {
  ENTER; W w = rec("vkCmdCopyBuffer");
  w_buffer_copy(w, "", cb, H(s), H(d), n, &r->srcOffset, &r->dstOffset, &r->size, sizeof(VkBufferCopy));
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer2(VkCommandBuffer cb, const VkCopyBufferInfo2* i) {
  ENTER; W w = rec("vkCmdCopyBuffer");
  w_buffer_copy(w, "", cb, H(i->srcBuffer), H(i->dstBuffer), i->regionCount, &i->pRegions->srcOffset,
                &i->pRegions->dstOffset, &i->pRegions->size, sizeof(VkBufferCopy2));
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout l, uint32_t n,
                                                  const VkBufferImageCopy* r) {
  ENTER; W w = rec("vkCmdCopyBufferToImage");
  w.k("cb").h(H(cb)).k("src").h(H(s)).k("dst").h(H(d)).k("dst_layout").en(name_VkImageLayout(l), l).k("regions").ab();
  for (uint32_t i = 0; i < n; i++) { w.ob().k("buffer_offset").u(r[i].bufferOffset).k("image"); w_layers(w, r[i].imageSubresource); w.oe(); }
  w.ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage2(VkCommandBuffer cb, const VkCopyBufferToImageInfo2* i) {
  ENTER; W w = rec("vkCmdCopyBufferToImage");
  w.k("cb").h(H(cb)).k("src").h(H(i->srcBuffer)).k("dst").h(H(i->dstImage))
   .k("dst_layout").en(name_VkImageLayout(i->dstImageLayout), i->dstImageLayout).k("regions").ab();
  for (uint32_t j = 0; j < i->regionCount; j++) {
    w.ob().k("buffer_offset").u(i->pRegions[j].bufferOffset).k("image"); w_layers(w, i->pRegions[j].imageSubresource); w.oe();
  }
  w.ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage s, VkImageLayout l, VkBuffer d, uint32_t n,
                                                  const VkBufferImageCopy* r) {
  ENTER; W w = rec("vkCmdCopyImageToBuffer");
  w.k("cb").h(H(cb)).k("src").h(H(s)).k("src_layout").en(name_VkImageLayout(l), l).k("dst").h(H(d)).k("regions").ab();
  for (uint32_t i = 0; i < n; i++) {
    w.ob().k("buffer_offset").u(r[i].bufferOffset).k("row_length").u(r[i].bufferRowLength)
     .k("image_height").u(r[i].bufferImageHeight)
     .k("extent").ab().u(r[i].imageExtent.width).u(r[i].imageExtent.height).u(r[i].imageExtent.depth).ae()
     .k("image"); w_layers(w, r[i].imageSubresource); w.oe();
    g_readops[H(cb)].push_back({H(s), H(d), r[i]});
  }
  w.ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer2(VkCommandBuffer cb, const VkCopyImageToBufferInfo2* i) {
  std::vector<VkBufferImageCopy> r(i->regionCount);
  for (uint32_t k = 0; k < i->regionCount; k++) {
    auto& s = i->pRegions[k];
    r[k] = {s.bufferOffset, s.bufferRowLength, s.bufferImageHeight, s.imageSubresource, s.imageOffset, s.imageExtent};
  }
  vkCmdCopyImageToBuffer(cb, i->srcImage, i->srcImageLayout, i->dstBuffer, i->regionCount, r.data());
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
                                          const VkImageCopy* r) {
  ENTER; W w = rec("vkCmdCopyImage");
  w.k("cb").h(H(cb)).k("src").h(H(s)).k("src_layout").en(name_VkImageLayout(sl), sl)
   .k("dst").h(H(d)).k("dst_layout").en(name_VkImageLayout(dl), dl).k("regions").ab();
  for (uint32_t i = 0; i < n; i++) { w.ob().k("src"); w_layers(w, r[i].srcSubresource); w.k("dst"); w_layers(w, r[i].dstSubresource); w.oe(); }
  w.ae(); emit(w);
}
static void blit(VkCommandBuffer cb, uint64_t s, VkImageLayout sl, uint64_t d, VkImageLayout dl, VkFilter f,
                 uint32_t n, const VkImageSubresourceLayers* ss, const VkImageSubresourceLayers* ds, size_t stride) {
  W w = rec("vkCmdBlitImage");
  w.k("cb").h(H(cb)).k("src").h(s).k("src_layout").en(name_VkImageLayout(sl), sl)
   .k("dst").h(d).k("dst_layout").en(name_VkImageLayout(dl), dl).k("filter").en(name_VkFilter(f), f).k("regions").ab();
  for (uint32_t i = 0; i < n; i++) {
    w.ob().k("src"); w_layers(w, *(const VkImageSubresourceLayers*)((const char*)ss + i * stride));
    w.k("dst"); w_layers(w, *(const VkImageSubresourceLayers*)((const char*)ds + i * stride)); w.oe();
  }
  w.ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
                                          const VkImageBlit* r, VkFilter f) {
  ENTER; blit(cb, H(s), sl, H(d), dl, f, n, &r->srcSubresource, &r->dstSubresource, sizeof(VkImageBlit));
}
VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage2(VkCommandBuffer cb, const VkBlitImageInfo2* i) {
  ENTER; blit(cb, H(i->srcImage), i->srcImageLayout, H(i->dstImage), i->dstImageLayout, i->filter, i->regionCount,
              &i->pRegions->srcSubresource, &i->pRegions->dstSubresource, sizeof(VkImageBlit2));
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage2(VkCommandBuffer cb, const VkCopyImageInfo2* i) {
  std::vector<VkImageCopy> r(i->regionCount);
  for (uint32_t k = 0; k < i->regionCount; k++) {
    auto& s = i->pRegions[k];
    r[k] = {s.srcSubresource, s.srcOffset, s.dstSubresource, s.dstOffset, s.extent};
  }
  vkCmdCopyImage(cb, i->srcImage, i->srcImageLayout, i->dstImage, i->dstImageLayout, i->regionCount, r.data());
}
VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize s, uint32_t) {
  ENTER; W w = rec("vkCmdFillBuffer"); w.k("cb").h(H(cb)).k("dst").h(H(b)).k("offset").u(o).k("size");
  if (s == VK_WHOLE_SIZE) w.str("whole"); else w.u(s);
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize s, const void*) {
  ENTER; W w = rec("vkCmdUpdateBuffer"); w.k("cb").h(H(cb)).k("dst").h(H(b)).k("offset").u(o).k("size").u(s); emit(w);
}
static void w_ranges(W& w, uint32_t n, const VkImageSubresourceRange* r) {
  w.k("ranges").ab(); for (uint32_t i = 0; i < n; i++) { w.ob(); w_range(w, r[i]); w.oe(); } w.ae();
}
VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue* v, uint32_t nr,
                                                const VkImageSubresourceRange* rg) {
  ENTER; W w = rec("vkCmdClearColorImage"); w.k("cb").h(H(cb)).k("dst").h(H(i)).k("dst_layout").en(name_VkImageLayout(l), l);
  w_ranges(w, nr, rg);
  w.k("color").ab().d(v->float32[0]).d(v->float32[1]).d(v->float32[2]).d(v->float32[3]).ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearDepthStencilValue* v,
                                                       uint32_t nr, const VkImageSubresourceRange* rg) {
  ENTER; W w = rec("vkCmdClearDepthStencilImage"); w.k("cb").h(H(cb)).k("dst").h(H(i)).k("dst_layout").en(name_VkImageLayout(l), l);
  w_ranges(w, nr, rg);
  w.k("depth").d(v->depth).k("stencil").u(v->stencil); emit(w);
}
// clears inside a render pass (GTK 4's Vulkan renderer) and query result copies (wgpu_mipmap)
VKAPI_ATTR void VKAPI_CALL vkCmdClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment* a, uint32_t nr,
                                                 const VkClearRect* r) {
  ENTER; W w = rec("vkCmdClearAttachments"); w.k("cb").h(H(cb)).k("attachments").ab();
  for (uint32_t i = 0; i < na; i++) {
    w.ob().k("aspect").fl(flags_VkImageAspectFlagBits, a[i].aspectMask).k("color").u(a[i].colorAttachment).oe();
  }
  w.ae().k("rects").ab();
  for (uint32_t i = 0; i < nr; i++) w.ob().k("base_layer").u(r[i].baseArrayLayer).k("layer_count").u(r[i].layerCount).oe();
  w.ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyQueryPoolResults(VkCommandBuffer cb, VkQueryPool, uint32_t first, uint32_t n, VkBuffer b,
                                                     VkDeviceSize o, VkDeviceSize stride, VkQueryResultFlags flags) {
  ENTER; W w = rec("vkCmdCopyQueryPoolResults"); w.k("cb").h(H(cb)).k("dst").h(H(b)).k("offset").u(o)
      .k("first").u(first).k("count").u(n).k("stride").u(stride).k("flags").u(flags); emit(w);
}
// no external handle types are supported (the model driver shares nothing outside the process)
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalSemaphoreProperties(VkPhysicalDevice, const VkPhysicalDeviceExternalSemaphoreInfo*,
                                                                          VkExternalSemaphoreProperties* p) {
  p->exportFromImportedHandleTypes = 0; p->compatibleHandleTypes = 0; p->externalSemaphoreFeatures = 0;
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalSemaphorePropertiesKHR(VkPhysicalDevice d, const VkPhysicalDeviceExternalSemaphoreInfo* i,
                                                                             VkExternalSemaphoreProperties* p) {
  vkGetPhysicalDeviceExternalSemaphoreProperties(d, i, p);
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalFenceProperties(VkPhysicalDevice, const VkPhysicalDeviceExternalFenceInfo*,
                                                                      VkExternalFenceProperties* p) {
  p->exportFromImportedHandleTypes = 0; p->compatibleHandleTypes = 0; p->externalFenceFeatures = 0;
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalFencePropertiesKHR(VkPhysicalDevice d, const VkPhysicalDeviceExternalFenceInfo* i,
                                                                         VkExternalFenceProperties* p) {
  vkGetPhysicalDeviceExternalFenceProperties(d, i, p);
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalBufferProperties(VkPhysicalDevice, const VkPhysicalDeviceExternalBufferInfo*,
                                                                       VkExternalBufferProperties* p) {
  p->externalMemoryProperties = VkExternalMemoryProperties{0, 0, 0};
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceExternalBufferPropertiesKHR(VkPhysicalDevice d, const VkPhysicalDeviceExternalBufferInfo* i,
                                                                          VkExternalBufferProperties* p) {
  vkGetPhysicalDeviceExternalBufferProperties(d, i, p);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthCompareOp(VkCommandBuffer cb, VkCompareOp op) {
  ENTER; W w = rec("vkCmdSetDepthCompareOp"); w.k("cb").h(H(cb)).k("op").en(name_VkCompareOp(op), op); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthTestEnable(VkCommandBuffer cb, VkBool32 e) {
  ENTER; W w = rec("vkCmdSetDepthTestEnable"); w.k("cb").h(H(cb)).k("enable").b(e); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthWriteEnable(VkCommandBuffer cb, VkBool32 e) {
  ENTER; W w = rec("vkCmdSetDepthWriteEnable"); w.k("cb").h(H(cb)).k("enable").b(e); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetStencilTestEnable(VkCommandBuffer cb, VkBool32 e) {   // stencil aspect accesses
  ENTER; W w = rec("vkCmdSetStencilTestEnable"); w.k("cb").h(H(cb)).k("enable").b(e); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdExecuteCommands(VkCommandBuffer cb, uint32_t n, const VkCommandBuffer* s) {
  ENTER; W w = rec("vkCmdExecuteCommands"); w.k("cb").h(H(cb)).k("command_buffers").ab();
  for (uint32_t i = 0; i < n; i++) w.h(H(s[i]));
  w.ae(); emit(w);
}

VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags ss, VkPipelineStageFlags ds, VkDependencyFlags df,
                                                uint32_t nm, const VkMemoryBarrier* m, uint32_t nb, const VkBufferMemoryBarrier* b,
                                                uint32_t ni, const VkImageMemoryBarrier* im) {
  ENTER;
  W w = rec("vkCmdPipelineBarrier");
  w.k("cb").h(H(cb)).k("dependency_flags").fl(flags_VkDependencyFlagBits, df);
  w.k("memory_barriers").ab();
  for (uint32_t i = 0; i < nm; i++) w_memory_barrier(w, ss, m[i].srcAccessMask, ds, m[i].dstAccessMask);
  w.ae().k("buffer_barriers").ab();
  for (uint32_t i = 0; i < nb; i++)
    w_buffer_barrier(w, H(b[i].buffer), ss, b[i].srcAccessMask, ds, b[i].dstAccessMask, b[i].srcQueueFamilyIndex,
                     b[i].dstQueueFamilyIndex, b[i].offset, b[i].size);
  w.ae().k("image_barriers").ab();
  for (uint32_t i = 0; i < ni; i++)
    w_image_barrier(w, H(im[i].image), im[i].oldLayout, im[i].newLayout, ss, im[i].srcAccessMask, ds, im[i].dstAccessMask,
                    im[i].srcQueueFamilyIndex, im[i].dstQueueFamilyIndex, im[i].subresourceRange);
  w.ae();
  w.k("src_stage"); stages(w, ss); w.k("dst_stage"); stages(w, ds);
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* di) {
  ENTER;
  W w = rec("vkCmdPipelineBarrier2");
  w.k("cb").h(H(cb)).k("dependency_flags").fl(flags_VkDependencyFlagBits, di->dependencyFlags);
  w.k("memory_barriers").ab();
  for (uint32_t i = 0; i < di->memoryBarrierCount; i++) {
    auto& m = di->pMemoryBarriers[i];
    w_memory_barrier(w, m.srcStageMask, m.srcAccessMask, m.dstStageMask, m.dstAccessMask);
  }
  w.ae().k("buffer_barriers").ab();
  for (uint32_t i = 0; i < di->bufferMemoryBarrierCount; i++) {
    auto& b = di->pBufferMemoryBarriers[i];
    w_buffer_barrier(w, H(b.buffer), b.srcStageMask, b.srcAccessMask, b.dstStageMask, b.dstAccessMask,
                     b.srcQueueFamilyIndex, b.dstQueueFamilyIndex, b.offset, b.size);
  }
  w.ae().k("image_barriers").ab();
  for (uint32_t i = 0; i < di->imageMemoryBarrierCount; i++) {
    auto& b = di->pImageMemoryBarriers[i];
    w_image_barrier(w, H(b.image), b.oldLayout, b.newLayout, b.srcStageMask, b.srcAccessMask, b.dstStageMask,
                    b.dstAccessMask, b.srcQueueFamilyIndex, b.dstQueueFamilyIndex, b.subresourceRange);
  }
  w.ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfo* di) { vkCmdPipelineBarrier2(cb, di); }

// ---------------------------------------------------------------- events (split barriers)
// vkCmdSetEvent* / vkCmdResetEvent* mark the point whose earlier commands form the
// first synchronization scope; vkCmdWaitEvents* carry the barriers (same keys as
// vkCmdPipelineBarrier*), applied per event by the loader
static void w_dep_info(W& w, const VkDependencyInfo* di) {
  w.ob().k("dependency_flags").fl(flags_VkDependencyFlagBits, di ? di->dependencyFlags : 0);
  w.k("memory_barriers").ab();
  for (uint32_t i = 0; di && i < di->memoryBarrierCount; i++) {
    auto& m = di->pMemoryBarriers[i];
    w_memory_barrier(w, m.srcStageMask, m.srcAccessMask, m.dstStageMask, m.dstAccessMask);
  }
  w.ae().k("buffer_barriers").ab();
  for (uint32_t i = 0; di && i < di->bufferMemoryBarrierCount; i++) {
    auto& b = di->pBufferMemoryBarriers[i];
    w_buffer_barrier(w, H(b.buffer), b.srcStageMask, b.srcAccessMask, b.dstStageMask, b.dstAccessMask,
                     b.srcQueueFamilyIndex, b.dstQueueFamilyIndex, b.offset, b.size);
  }
  w.ae().k("image_barriers").ab();
  for (uint32_t i = 0; di && i < di->imageMemoryBarrierCount; i++) {
    auto& b = di->pImageMemoryBarriers[i];
    w_image_barrier(w, H(b.image), b.oldLayout, b.newLayout, b.srcStageMask, b.srcAccessMask, b.dstStageMask,
                    b.dstAccessMask, b.srcQueueFamilyIndex, b.dstQueueFamilyIndex, b.subresourceRange);
  }
  w.ae().oe();
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent(VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags sm) {
  ENTER; W w = rec("vkCmdSetEvent"); w.k("cb").h(H(cb)).k("event").h(H(e)).k("stage"); stages(w, sm); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent2(VkCommandBuffer cb, VkEvent e, const VkDependencyInfo* di) {
  ENTER; W w = rec("vkCmdSetEvent2"); w.k("cb").h(H(cb)).k("event").h(H(e)).k("dependency"); w_dep_info(w, di); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdSetEvent2KHR(VkCommandBuffer cb, VkEvent e, const VkDependencyInfo* di) { vkCmdSetEvent2(cb, e, di); }
VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent(VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags sm) {
  ENTER; W w = rec("vkCmdResetEvent"); w.k("cb").h(H(cb)).k("event").h(H(e)).k("stage"); stages(w, sm); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent2(VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags2 sm) {
  ENTER; W w = rec("vkCmdResetEvent2"); w.k("cb").h(H(cb)).k("event").h(H(e)).k("stage"); stages(w, sm); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdResetEvent2KHR(VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags2 sm) { vkCmdResetEvent2(cb, e, sm); }
VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents(VkCommandBuffer cb, uint32_t ne, const VkEvent* ev, VkPipelineStageFlags ss,
                                           VkPipelineStageFlags ds, uint32_t nm, const VkMemoryBarrier* m, uint32_t nb,
                                           const VkBufferMemoryBarrier* b, uint32_t ni, const VkImageMemoryBarrier* im) {
  ENTER;
  W w = rec("vkCmdWaitEvents");
  w.k("cb").h(H(cb)).k("events").ab();
  for (uint32_t i = 0; i < ne; i++) w.h(H(ev[i]));
  w.ae().k("memory_barriers").ab();
  for (uint32_t i = 0; i < nm; i++) w_memory_barrier(w, ss, m[i].srcAccessMask, ds, m[i].dstAccessMask);
  w.ae().k("buffer_barriers").ab();
  for (uint32_t i = 0; i < nb; i++)
    w_buffer_barrier(w, H(b[i].buffer), ss, b[i].srcAccessMask, ds, b[i].dstAccessMask, b[i].srcQueueFamilyIndex,
                     b[i].dstQueueFamilyIndex, b[i].offset, b[i].size);
  w.ae().k("image_barriers").ab();
  for (uint32_t i = 0; i < ni; i++)
    w_image_barrier(w, H(im[i].image), im[i].oldLayout, im[i].newLayout, ss, im[i].srcAccessMask, ds, im[i].dstAccessMask,
                    im[i].srcQueueFamilyIndex, im[i].dstQueueFamilyIndex, im[i].subresourceRange);
  w.ae();
  w.k("src_stage"); stages(w, ss); w.k("dst_stage"); stages(w, ds);
  emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents2(VkCommandBuffer cb, uint32_t ne, const VkEvent* ev, const VkDependencyInfo* di) {
  ENTER;
  W w = rec("vkCmdWaitEvents2");
  w.k("cb").h(H(cb)).k("events").ab();
  for (uint32_t i = 0; i < ne; i++) w.h(H(ev[i]));
  w.ae().k("dependencies").ab();
  for (uint32_t i = 0; i < ne; i++) w_dep_info(w, &di[i]);
  w.ae(); emit(w);
}
VKAPI_ATTR void VKAPI_CALL vkCmdWaitEvents2KHR(VkCommandBuffer cb, uint32_t ne, const VkEvent* ev, const VkDependencyInfo* di) {
  vkCmdWaitEvents2(cb, ne, ev, di);
}
// host-side event operations: recorded (the loader reports host-signalled events)
VKAPI_ATTR VkResult VKAPI_CALL vkSetEvent(VkDevice, VkEvent e) {
  ENTER; W w = rec("vkSetEvent"); w.k("event").h(H(e)); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetEvent(VkDevice, VkEvent e) {
  ENTER; W w = rec("vkResetEvent"); w.k("event").h(H(e)); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetEventStatus(VkDevice, VkEvent e) {
  ENTER; W w = rec("vkGetEventStatus"); w.k("event").h(H(e)).k("result").str("VK_EVENT_SET"); emit(w); return VK_EVENT_SET;
}

// ---------------------------------------------------------------- sync and submission
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(VkDevice, const VkSemaphoreCreateInfo* ci, const VkAllocationCallbacks*, VkSemaphore* out) {
  ENTER;
  uint64_t h = mk("semaphore"); *out = as<VkSemaphore>(h);
  auto* t = find_chain<VkSemaphoreTypeCreateInfo>(ci->pNext, VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
  bool tl = t && t->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE;
  W w = rec("vkCreateSemaphore");
  w.k("semaphore").h(h).k("type").str(tl ? "timeline" : "binary");
  if (tl) w.k("initial_value").u(t->initialValue);
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkSignalSemaphore(VkDevice, const VkSemaphoreSignalInfo* i) {
  ENTER; W w = rec("vkSignalSemaphore"); w.k("semaphore").h(H(i->semaphore)).k("value").u(i->value); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkWaitSemaphores(VkDevice, const VkSemaphoreWaitInfo* i, uint64_t timeout) {
  ENTER; W w = rec("vkWaitSemaphores");
  w.k("wait_any").b(i->flags & VK_SEMAPHORE_WAIT_ANY_BIT).k("semaphores").ab();
  for (uint32_t k = 0; k < i->semaphoreCount; k++) w.ob().k("semaphore").h(H(i->pSemaphores[k])).k("value").u(i->pValues[k]).oe();
  w.ae().k("timeout");
  if (timeout == UINT64_MAX) w.str("infinite"); else w.u(timeout);
  emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetSemaphoreCounterValue(VkDevice, VkSemaphore s, uint64_t* v) {
  ENTER; *v = UINT64_MAX >> 1;
  // record the value returned, so the checker can treat the query as a host wait for it
  W w = rec("vkGetSemaphoreCounterValue"); w.k("semaphore").h(H(s)).k("value").u(*v); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice, const VkFenceCreateInfo* ci, const VkAllocationCallbacks*, VkFence* out) {
  ENTER;
  uint64_t h = mk("fence"); *out = as<VkFence>(h);
  W w = rec("vkCreateFence"); w.k("fence").h(h).k("signaled").b(ci->flags & VK_FENCE_CREATE_SIGNALED_BIT); emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice, uint32_t n, const VkFence* f, VkBool32 all, uint64_t timeout) {
  ENTER;
  W w = rec("vkWaitForFences"); w.k("fences").ab();
  for (uint32_t i = 0; i < n; i++) w.h(H(f[i]));
  w.ae().k("wait_all").b(all).k("timeout");
  if (timeout == UINT64_MAX) w.str("infinite"); else w.u(timeout);
  emit(w);
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice, uint32_t n, const VkFence* f) {
  ENTER; W w = rec("vkResetFences"); w.k("fences").ab();
  for (uint32_t i = 0; i < n; i++) w.h(H(f[i]));
  w.ae(); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice, VkFence f) {
  ENTER; W w = rec("vkGetFenceStatus"); w.k("fence").h(H(f)).k("result").str("VK_SUCCESS"); emit(w); return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateEvent(VkDevice, const VkEventCreateInfo*, const VkAllocationCallbacks*, VkEvent* out) {
  ENTER; uint64_t h = mk("event"); *out = as<VkEvent>(h);
  W w = rec("vkCreateEvent"); w.k("event").h(h); emit(w); return VK_SUCCESS;
}

// --- readback probe: an image -> host buffer copy has no GPU to produce its
// data, so at submit the destination gets a ramp in the image's format: the
// texel in column c holds RAMP[c % 48] in R, G and B (alpha 1), with
// RAMP = 16 log-spaced values in [1e-3, 1e-1) then 32 evenly spaced in
// [0.1, 1] (covenant/transfer.py RAMP). A covenant_readback_fill record says
// where; the meaning checker fits what the host makes of these values (e.g.
// the bytes it writes to an image file) against the transfer library.
float ramp_value(uint32_t c) {
  uint32_t i = c % 48;
  if (i < 16) return (float)pow(10.0, -3.0 + 2.0 * i / 16.0);
  return (float)(0.1 + 0.9 * (i - 16) / 31.0);
}
uint16_t to_half(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000; int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
  if (e <= 0) { if (e < -10) return (uint16_t)sign; m |= 0x800000; return (uint16_t)(sign | (m >> (14 - e))); }
  if (e >= 31) return (uint16_t)(sign | 0x7c00);
  return (uint16_t)(sign | (e << 10) | (m >> 13));
}
bool encode_texel(VkFormat f, float v, uint8_t* out, uint32_t* size) {
  uint8_t b = (uint8_t)lround(fmin(fmax(v, 0.f), 1.f) * 255.f);
  switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
      out[0] = out[1] = out[2] = b; out[3] = 255; *size = 4; return true;
    case VK_FORMAT_R16G16B16A16_SFLOAT: {
      uint16_t h = to_half(v), one = to_half(1.f);
      memcpy(out, &h, 2); memcpy(out + 2, &h, 2); memcpy(out + 4, &h, 2); memcpy(out + 6, &one, 2);
      *size = 8; return true;
    }
    case VK_FORMAT_R32G32B32A32_SFLOAT: {
      float one = 1.f; memcpy(out, &v, 4); memcpy(out + 4, &v, 4); memcpy(out + 8, &v, 4); memcpy(out + 12, &one, 4);
      *size = 16; return true;
    }
    default: return false;
  }
}
void readback_fill(uint64_t cb) {
  static int off = -1;
  if (off < 0) { const char* e = getenv("COVENANT_NO_READBACK_FILL"); off = (e && *e && *e != '0') ? 1 : 0; }
  if (off) return;
  auto it = g_readops.find(cb);
  if (it == g_readops.end()) return;
  for (auto& op : it->second) {
    auto ii = g_image.find(op.image); auto bi = g_bufbind.find(op.buffer);
    if (ii == g_image.end() || bi == g_bufbind.end()) continue;
    auto mi = g_mem.find(bi->second.first);
    if (mi == g_mem.end() || !mi->second.base) continue;
    MemObj& m = mi->second;
    uint8_t texel[16]; uint32_t ts = 0;
    VkFormat fmt = ii->second.format;
    if (!encode_texel(fmt, 0.5f, texel, &ts)) continue;
    const VkBufferImageCopy& r = op.region;
    uint32_t w = r.imageExtent.width, h = r.imageExtent.height;
    uint32_t rowlen = r.bufferRowLength ? r.bufferRowLength : w;
    size_t base = bi->second.second + r.bufferOffset;
    size_t span = ((size_t)(h ? h - 1 : 0) * rowlen + w) * ts;
    if (!w || !h || base + span > m.len) continue;
    if (m.region >= 0) mprotect(m.base, m.len, PROT_READ | PROT_WRITE);
    for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++) {
        encode_texel(fmt, ramp_value(x), texel, &ts);
        memcpy(m.base + base + ((size_t)y * rowlen + x) * ts, texel, ts);
      }
    if (m.region >= 0) {
      Region& rg = g_reg[m.region];
      if (rg.shadow) memcpy(rg.shadow + base, m.base + base, span);   // a device write, not a host one
      mprotect(m.base, m.len, PROT_READ);
    }
    W wr = rec("covenant_readback_fill");
    wr.k("cb").h(cb).k("image").h(op.image).k("buffer").h(op.buffer).k("memory").h(bi->second.first)
      .k("offset").u(base).k("format").en(name_VkFormat(fmt), fmt).k("texel_size").u(ts)
      .k("extent").ab().u(w).u(h).ae().k("row_length").u(rowlen).k("ramp_period").u(48);
    emit(wr);
  }
}

// The instances a top-level build reads are device data. When the instance
// buffer lives in host-visible memory the app has mapped, read its
// accelerationStructureReference fields at submit (the GPU reads them when
// the build executes, after the submit): one covenant_tlas_refs record per
// TLAS build of the command buffer, emitted before the submit record.
// Otherwise "resolved": false (host_trace then assumes every live BLAS).
void tlas_refs(uint64_t cb) {
  auto it = g_tlas_pending.find(cb);
  if (it == g_tlas_pending.end()) return;
  for (auto& p : it->second) {
    W w = rec("covenant_tlas_refs");
    w.k("cb").h(cb).k("cmd_seq").u(p.seq).k("build").u(p.build);
    const uint8_t* base = nullptr;
    auto bi = g_bufbind.find(p.buffer);
    if (bi != g_bufbind.end()) {
      auto mi = g_mem.find(bi->second.first);
      if (mi != g_mem.end() && mi->second.base &&
          bi->second.second + p.offset + (VkDeviceSize)p.count * (p.pointers ? 8 : 64) <= mi->second.len)
        base = (const uint8_t*)mi->second.base + bi->second.second + p.offset;
    }
    if (!base || p.pointers) { w.k("resolved").b(false); emit(w); continue; }
    w.k("resolved").b(true).k("refs").ab();
    std::set<uint64_t> seen;
    for (uint32_t k = 0; k < p.count; k++) {
      uint64_t ref; memcpy(&ref, base + 64 * k + 56, 8);
      if (!ref || !seen.insert(ref).second) continue;
      auto a = g_asaddr.find(ref);
      if (a != g_asaddr.end()) w.h(a->second);
      else { char b[40]; snprintf(b, 40, "address#%llx", (unsigned long long)ref); w.str(b); }
    }
    w.ae(); emit(w);
  }
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* s, VkFence f) {
  ENTER;
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < s[i].commandBufferCount; j++) tlas_refs(H(s[i].pCommandBuffers[j]));
  W w = rec("vkQueueSubmit");
  w.k("queue").h(H(q)).k("submits").ab();
  for (uint32_t i = 0; i < n; i++) {
    auto* tl = find_chain<VkTimelineSemaphoreSubmitInfo>(s[i].pNext, VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
    w.ob();
    w_sem_infos(w, "wait", s[i].waitSemaphoreCount, s[i].pWaitSemaphores, tl ? tl->pWaitSemaphoreValues : nullptr,
                tl ? tl->waitSemaphoreValueCount : 0, s[i].pWaitDstStageMask);
    w.k("command_buffers").ab();
    for (uint32_t j = 0; j < s[i].commandBufferCount; j++) w.h(H(s[i].pCommandBuffers[j]));
    w.ae();
    w_sem_infos(w, "signal", s[i].signalSemaphoreCount, s[i].pSignalSemaphores, tl ? tl->pSignalSemaphoreValues : nullptr,
                tl ? tl->signalSemaphoreValueCount : 0, nullptr);
    w.oe();
  }
  w.ae().k("fence").h(H(f));
  emit(w);
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < s[i].commandBufferCount; j++) readback_fill(H(s[i].pCommandBuffers[j]));
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) {
  ENTER;
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < s[i].commandBufferInfoCount; j++) tlas_refs(H(s[i].pCommandBufferInfos[j].commandBuffer));
  W w = rec("vkQueueSubmit");          // same schema as vkQueueSubmit
  w.k("queue").h(H(q)).k("submit2").b(true).k("submits").ab();
  for (uint32_t i = 0; i < n; i++) {
    w.ob();
    w_sem_infos2(w, "wait", s[i].waitSemaphoreInfoCount, s[i].pWaitSemaphoreInfos);
    w.k("command_buffers").ab();
    for (uint32_t j = 0; j < s[i].commandBufferInfoCount; j++) w.h(H(s[i].pCommandBufferInfos[j].commandBuffer));
    w.ae();
    w_sem_infos2(w, "signal", s[i].signalSemaphoreInfoCount, s[i].pSignalSemaphoreInfos);
    w.oe();
  }
  w.ae().k("fence").h(H(f));
  emit(w);
  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < s[i].commandBufferInfoCount; j++) readback_fill(H(s[i].pCommandBufferInfos[j].commandBuffer));
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit2KHR(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) { return vkQueueSubmit2(q, n, s, f); }

// ---------------------------------------------------------------- destroy
#define COVVK_DESTROY(Name, Parent, Type)                                                       \
  VKAPI_ATTR void VKAPI_CALL vkDestroy##Name(Parent, Type h, const VkAllocationCallbacks*) {    \
    ENTER; if (!h) return;                                                                       \
    W w = rec("vkDestroy" #Name); w.k("handle").h(H(h)); emit(w);                               \
  }
COVVK_DESTROY(SurfaceKHR, VkInstance, VkSurfaceKHR)
COVVK_DESTROY(DebugUtilsMessengerEXT, VkInstance, VkDebugUtilsMessengerEXT)
COVVK_DESTROY(SwapchainKHR, VkDevice, VkSwapchainKHR)
COVVK_DESTROY(Image, VkDevice, VkImage)
COVVK_DESTROY(ImageView, VkDevice, VkImageView)
COVVK_DESTROY(Buffer, VkDevice, VkBuffer)
COVVK_DESTROY(BufferView, VkDevice, VkBufferView)
COVVK_DESTROY(Sampler, VkDevice, VkSampler)
COVVK_DESTROY(ShaderModule, VkDevice, VkShaderModule)
COVVK_DESTROY(DescriptorSetLayout, VkDevice, VkDescriptorSetLayout)
COVVK_DESTROY(DescriptorPool, VkDevice, VkDescriptorPool)
COVVK_DESTROY(PipelineLayout, VkDevice, VkPipelineLayout)
COVVK_DESTROY(Pipeline, VkDevice, VkPipeline)
COVVK_DESTROY(PipelineCache, VkDevice, VkPipelineCache)
COVVK_DESTROY(RenderPass, VkDevice, VkRenderPass)
COVVK_DESTROY(Framebuffer, VkDevice, VkFramebuffer)
COVVK_DESTROY(CommandPool, VkDevice, VkCommandPool)
COVVK_DESTROY(Semaphore, VkDevice, VkSemaphore)
COVVK_DESTROY(Fence, VkDevice, VkFence)
COVVK_DESTROY(Event, VkDevice, VkEvent)
COVVK_DESTROY(QueryPool, VkDevice, VkQueryPool)

}  // extern "C"

// ---------------------------------------------------------------- generated stubs + proc table
#include "covvk_stubs.inc"

// commands of platform headers (not in vulkan_core.h, so not in covvk_procs)
static const ProcEntry covvk_platform_procs[] = {
  {"vkCreateXcbSurfaceKHR", (PFN_vkVoidFunction)&vkCreateXcbSurfaceKHR},
  {"vkCreateXlibSurfaceKHR", (PFN_vkVoidFunction)&vkCreateXlibSurfaceKHR},
  {"vkCreateWaylandSurfaceKHR", (PFN_vkVoidFunction)&vkCreateWaylandSurfaceKHR},
  {"vkGetPhysicalDeviceXcbPresentationSupportKHR", (PFN_vkVoidFunction)&vkGetPhysicalDeviceXcbPresentationSupportKHR},
  {"vkGetPhysicalDeviceXlibPresentationSupportKHR", (PFN_vkVoidFunction)&vkGetPhysicalDeviceXlibPresentationSupportKHR},
  {"vkGetPhysicalDeviceWaylandPresentationSupportKHR", (PFN_vkVoidFunction)&vkGetPhysicalDeviceWaylandPresentationSupportKHR},
  {nullptr, nullptr}};

static PFN_vkVoidFunction lookup(const char* name) {
  if (!name) return nullptr;
  for (const ProcEntry* e = covvk_platform_procs; e->name; e++)
    if (!strcmp(e->name, name)) return e->fn;
  for (const ProcEntry* e = covvk_procs; e->name; e++)
    if (!strcmp(e->name, name)) return e->fn;
  // KHR/EXT aliases of promoted commands: vkCmdBeginRenderingKHR -> vkCmdBeginRendering
  std::string n(name);
  for (const char* suf : {"KHR", "EXT"}) {
    size_t l = strlen(suf);
    if (n.size() > l && n.compare(n.size() - l, l, suf) == 0) {
      std::string base = n.substr(0, n.size() - l);
      for (const ProcEntry* e = covvk_procs; e->name; e++)
        if (base == e->name) return e->fn;
    }
  }
  return nullptr;
}

extern "C" {
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance, const char* name) { return lookup(name); }
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice, const char* name) { return lookup(name); }
}
