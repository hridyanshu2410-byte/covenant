// VK_LAYER_COVENANT_elide -- applies a COVENANT barrier-elision plan.
//
// The plan (covenant/elide.py) says, for recorded vkCmdPipelineBarrier /
// vkCmdPipelineBarrier2 calls, which barrier structs to drop and which to keep
// with narrower stage/access masks. This layer intercepts exactly those
// commands plus vkBeginCommandBuffer (to know which recording a call belongs
// to) and forwards everything else untouched (the loader calls the next layer
// directly for every other function: we never appear in its dispatch chain).
//
// Environment:
//   COVENANT_ELIDE_PLAN=<plan.json>   the plan; without it the layer is a pass-through
//   COVENANT_ELIDE_MODE=strict|chain  match by (recording ordinal, barrier index)
//                                     [default], or by the barrier-sequence chain hash
//   COVENANT_ELIDE_DISABLE=1          load but apply nothing (measures layer overhead)
//   COVENANT_ELIDE_STATS=<file>       write counters as JSON at exit
//   COVENANT_ELIDE_VERBOSE=1          print every action taken
//   COVENANT_ELIDE_QUIET=1            no summary line on stderr at exit
//
// Safety: a call is changed only when the plan has an entry for its key AND
// the entry's content signature equals the signature of the call actually
// being recorded (same masks, layouts, families, ranges, counts). Anything
// else is forwarded exactly as recorded.
#include <vulkan/vulkan.h>
#if defined(__has_include)
#if __has_include(<vulkan/vk_layer.h>)
#include <vulkan/vk_layer.h>
#define HAVE_VK_LAYER_H 1
#endif
#endif

#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

#ifndef HAVE_VK_LAYER_H
// ABI-stable loader/layer interface structs (from vk_layer.h), for SDKs that no longer ship it.
typedef enum VkLayerFunction_ { VK_LAYER_LINK_INFO = 0 } VkLayerFunction;
typedef PFN_vkVoidFunction(VKAPI_PTR* PFN_GetPhysicalDeviceProcAddr)(VkInstance, const char*);
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
  union { VkLayerDeviceLink* pLayerInfo; void* other; } u;
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

// ------------------------------------------------------------------ tiny JSON
struct J {
  enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<J> a;
  std::vector<std::pair<std::string, J>> o;
  const J* get(const char* k) const {
    for (auto& kv : o) if (kv.first == k) return &kv.second;
    return nullptr;
  }
  std::string str(const char* k, const char* def = "") const { auto* v = get(k); return v && v->t == STR ? v->s : def; }
  double num(const char* k, double def = 0) const { auto* v = get(k); return v && v->t == NUM ? v->n : def; }
};

struct Parser {
  const char* p; const char* e; bool ok = true;
  void ws() { while (p < e && isspace((unsigned char)*p)) p++; }
  bool lit(const char* s) { size_t n = strlen(s); if ((size_t)(e - p) >= n && !strncmp(p, s, n)) { p += n; return true; } return false; }
  J val() {
    J v; ws();
    if (p >= e) { ok = false; return v; }
    if (*p == '{') {
      v.t = J::OBJ; p++; ws();
      if (p < e && *p == '}') { p++; return v; }
      while (ok) {
        ws(); J k = val(); ws();
        if (k.t != J::STR || p >= e || *p != ':') { ok = false; break; }
        p++;
        v.o.emplace_back(k.s, val()); ws();
        if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == '}') { p++; break; }
        ok = false;
      }
    } else if (*p == '[') {
      v.t = J::ARR; p++; ws();
      if (p < e && *p == ']') { p++; return v; }
      while (ok) {
        v.a.push_back(val()); ws();
        if (p < e && *p == ',') { p++; continue; }
        if (p < e && *p == ']') { p++; break; }
        ok = false;
      }
    } else if (*p == '"') {
      v.t = J::STR; p++;
      while (p < e && *p != '"') {
        if (*p == '\\' && p + 1 < e) {
          p++;
          char c = *p++;
          switch (c) {
            case 'n': v.s += '\n'; break; case 't': v.s += '\t'; break; case 'r': v.s += '\r'; break;
            case 'b': v.s += '\b'; break; case 'f': v.s += '\f'; break;
            case 'u': { if (e - p >= 4) { unsigned cp = (unsigned)strtoul(std::string(p, 4).c_str(), nullptr, 16); p += 4;
                          if (cp < 0x80) v.s += (char)cp; else v.s += '?'; } break; }
            default: v.s += c;
          }
        } else v.s += *p++;
      }
      if (p < e) p++; else ok = false;
    } else if (lit("true")) { v.t = J::BOOL; v.b = true; }
    else if (lit("false")) { v.t = J::BOOL; }
    else if (lit("null")) { v.t = J::NUL; }
    else { char* end; v.t = J::NUM; v.n = strtod(p, &end); if (end == p) ok = false; p = end; }
    return v;
  }
};

// ------------------------------------------------------------------ the plan
enum Action : uint8_t { KEEP, REMOVE, NARROW };
struct ElemPlan { char kind; uint32_t index; Action action; uint64_t ss = 0, sa = 0, ds = 0, da = 0; };
struct Entry {
  uint64_t sig = 0;
  bool remove = false;
  uint64_t sync1_ss = 0, sync1_ds = 0;
  bool has_sync1 = false;
  std::vector<ElemPlan> elems;
  const ElemPlan* find(char kind, uint32_t i) const {
    for (auto& e : elems) if (e.kind == kind && e.index == i) return &e;
    return nullptr;
  }
};

uint64_t hex(const std::string& s) { return strtoull(s.c_str(), nullptr, 16); }

bool parse_entry(const J& b, Entry& out) {
  std::string d = b.str("decision");
  if (d != "remove" && d != "narrow") return false;
  out.sig = hex(b.str("sig"));
  out.remove = d == "remove";
  if (b.get("sync1_src_stage_mask")) {
    out.has_sync1 = true;
    out.sync1_ss = hex(b.str("sync1_src_stage_mask"));
    out.sync1_ds = hex(b.str("sync1_dst_stage_mask"));
  }
  if (auto* el = b.get("elements"))
    for (auto& e : el->a) {
      ElemPlan p{};
      std::string k = e.str("kind");
      p.kind = k == "memory" ? 'M' : k == "buffer" ? 'B' : k == "image" ? 'I' : 'E';
      p.index = (uint32_t)e.num("index");
      std::string a = e.str("action");
      p.action = a == "remove" ? REMOVE : a == "narrow" ? NARROW : KEEP;
      if (p.action == NARROW) {
        p.ss = hex(e.str("src_stage_mask")); p.sa = hex(e.str("src_access_mask"));
        p.ds = hex(e.str("dst_stage_mask")); p.da = hex(e.str("dst_access_mask"));
      }
      out.elems.push_back(p);
    }
  return true;
}

struct Stats {
  std::atomic<uint64_t> recordings{0}, calls{0}, matched{0}, sig_mismatch{0}, removed_calls{0}, narrowed_calls{0},
      structs_removed{0}, structs_narrowed{0};
};

struct Global {
  std::mutex mu;
  bool loaded = false, active = false, chain_mode = false, verbose = false, quiet = false;
  std::string plan_path, stats_path, mode = "strict";
  std::unordered_map<uint64_t, Entry> strict;   // (rec << 32) | idx
  std::unordered_map<uint64_t, Entry> chain;    // chain hash
  std::unordered_map<void*, PFN_vkGetInstanceProcAddr> inst_gipa;   // instance dispatch key -> next GIPA
  struct Dev {
    PFN_vkGetDeviceProcAddr gdpa = nullptr;
    PFN_vkDestroyDevice destroy = nullptr;
    PFN_vkBeginCommandBuffer begin = nullptr;
    PFN_vkCmdPipelineBarrier barrier1 = nullptr;
    PFN_vkCmdPipelineBarrier2 barrier2 = nullptr;
    PFN_vkCmdPipelineBarrier2 barrier2khr = nullptr;
  };
  std::unordered_map<void*, Dev> devs;           // device dispatch key -> next functions
  struct Cb { uint64_t rec = 0; uint32_t idx = 0; uint64_t chain = 0; };
  std::unordered_map<VkCommandBuffer, Cb> cbs;
  uint64_t next_rec = 0;
  Stats st;
};
Global& G() { static Global* g = new Global(); return *g; }   // never destroyed: safe at exit

void* key(const void* dispatchable) { return *(void**)dispatchable; }

void vlog(const char* fmt, ...) {
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "[covenant-elide] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

void write_stats() {
  Global& g = G();
  auto& s = g.st;
  char buf[1024];
  // JSON-escape the plan path (Windows paths contain backslashes).
  std::string plan_json;
  for (char c : g.plan_path) {
    if (c == '\\' || c == '"') plan_json += '\\';
    plan_json += c;
  }
  snprintf(buf, sizeof buf,
           "{\"plan\": \"%s\", \"mode\": \"%s\", \"active\": %s, \"recordings\": %llu, \"barrier_calls\": %llu, "
           "\"matched\": %llu, \"sig_mismatch\": %llu, \"removed_calls\": %llu, \"narrowed_calls\": %llu, "
           "\"structs_removed\": %llu, \"structs_narrowed\": %llu}",
           plan_json.c_str(), g.mode.c_str(), g.active ? "true" : "false", (unsigned long long)s.recordings.load(),
           (unsigned long long)s.calls.load(), (unsigned long long)s.matched.load(), (unsigned long long)s.sig_mismatch.load(),
           (unsigned long long)s.removed_calls.load(), (unsigned long long)s.narrowed_calls.load(),
           (unsigned long long)s.structs_removed.load(), (unsigned long long)s.structs_narrowed.load());
  if (!g.quiet) vlog("%s", buf);
  if (!g.stats_path.empty()) {
    FILE* f = fopen(g.stats_path.c_str(), "w");
    if (f) { fprintf(f, "%s\n", buf); fclose(f); }
  }
}

void load_plan() {
  Global& g = G();
  std::lock_guard<std::mutex> lk(g.mu);
  if (g.loaded) return;
  g.loaded = true;
  auto env = [](const char* n) { const char* v = getenv(n); return v ? std::string(v) : std::string(); };
  g.plan_path = env("COVENANT_ELIDE_PLAN");
  g.stats_path = env("COVENANT_ELIDE_STATS");
  g.verbose = env("COVENANT_ELIDE_VERBOSE") == "1";
  g.quiet = env("COVENANT_ELIDE_QUIET") == "1";
  if (!env("COVENANT_ELIDE_MODE").empty()) g.mode = env("COVENANT_ELIDE_MODE");
  g.chain_mode = g.mode == "chain";
  atexit(write_stats);
  if (g.plan_path.empty() || env("COVENANT_ELIDE_DISABLE") == "1") {
    if (!g.quiet) vlog("no plan applied (%s)", g.plan_path.empty() ? "COVENANT_ELIDE_PLAN not set" : "COVENANT_ELIDE_DISABLE=1");
    return;
  }
  FILE* f = fopen(g.plan_path.c_str(), "rb");
  if (!f) { vlog("cannot open plan %s; pass-through", g.plan_path.c_str()); return; }
  std::string text;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  fclose(f);
  Parser ps{text.data(), text.data() + text.size()};
  J root = ps.val();
  if (!ps.ok || root.t != J::OBJ || root.str("format") != "covenant-elide-plan/1") {
    vlog("plan %s is not a covenant-elide-plan/1 file; pass-through", g.plan_path.c_str());
    return;
  }
  if (auto* bs = root.get("barriers"))
    for (auto& b : bs->a) {
      Entry e;
      if (parse_entry(b, e)) g.strict[((uint64_t)b.num("rec") << 32) | (uint64_t)b.num("idx")] = std::move(e);
    }
  if (auto* cs = root.get("chain"))
    for (auto& b : cs->a) {
      Entry e;
      if (parse_entry(b, e)) g.chain[hex(b.str("chain"))] = std::move(e);
    }
  g.active = true;
  if (!g.quiet)
    vlog("plan %s: %zu strict entries, %zu chain entries; matching by %s", g.plan_path.c_str(), g.strict.size(),
         g.chain.size(), g.chain_mode ? "chain" : "strict (recording ordinal, barrier index)");
}

// ------------------------------------------------------------------ signatures (must match covenant/elide.py)
constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull, FNV_PRIME = 0x100000001b3ull;
uint64_t fnv(const std::string& s, uint64_t h = FNV_OFFSET) {
  for (unsigned char c : s) { h ^= c; h *= FNV_PRIME; }
  return h;
}

struct Sig {
  std::string s;
  void u(uint64_t v) { char b[24]; snprintf(b, sizeof b, "%llu", (unsigned long long)v); s += b; }
  void c(char ch) { s += ch; }
};

std::string sig1(VkPipelineStageFlags ss, VkPipelineStageFlags ds, VkDependencyFlags df, uint32_t nm, const VkMemoryBarrier* m,
                 uint32_t nb, const VkBufferMemoryBarrier* b, uint32_t ni, const VkImageMemoryBarrier* im) {
  Sig g; g.s = "1|"; g.u(df);
  for (uint32_t i = 0; i < nm; i++) {
    g.s += "|M:"; g.u(ss); g.c(':'); g.u(m[i].srcAccessMask); g.c(':'); g.u(ds); g.c(':'); g.u(m[i].dstAccessMask);
  }
  for (uint32_t i = 0; i < nb; i++) {
    g.s += "|B:"; g.u(ss); g.c(':'); g.u(b[i].srcAccessMask); g.c(':'); g.u(ds); g.c(':'); g.u(b[i].dstAccessMask);
    g.c(':'); g.u(b[i].srcQueueFamilyIndex); g.c(':'); g.u(b[i].dstQueueFamilyIndex); g.c(':'); g.u(b[i].offset); g.c(':'); g.u(b[i].size);
  }
  for (uint32_t i = 0; i < ni; i++) {
    auto& x = im[i];
    g.s += "|I:"; g.u(ss); g.c(':'); g.u(x.srcAccessMask); g.c(':'); g.u(ds); g.c(':'); g.u(x.dstAccessMask);
    g.c(':'); g.u(x.srcQueueFamilyIndex); g.c(':'); g.u(x.dstQueueFamilyIndex); g.c(':'); g.u((uint32_t)x.oldLayout);
    g.c(':'); g.u((uint32_t)x.newLayout); g.c(':'); g.u(x.subresourceRange.aspectMask); g.c(':'); g.u(x.subresourceRange.baseMipLevel);
    g.c(':'); g.u(x.subresourceRange.levelCount); g.c(':'); g.u(x.subresourceRange.baseArrayLayer); g.c(':'); g.u(x.subresourceRange.layerCount);
  }
  if (!nm && !nb && !ni) { g.s += "|E:"; g.u(ss); g.c(':'); g.u(ds); }
  return g.s;
}

std::string sig2(const VkDependencyInfo* di) {
  Sig g; g.s = "2|"; g.u(di->dependencyFlags);
  for (uint32_t i = 0; i < di->memoryBarrierCount; i++) {
    auto& m = di->pMemoryBarriers[i];
    g.s += "|M:"; g.u(m.srcStageMask); g.c(':'); g.u(m.srcAccessMask); g.c(':'); g.u(m.dstStageMask); g.c(':'); g.u(m.dstAccessMask);
  }
  for (uint32_t i = 0; i < di->bufferMemoryBarrierCount; i++) {
    auto& b = di->pBufferMemoryBarriers[i];
    g.s += "|B:"; g.u(b.srcStageMask); g.c(':'); g.u(b.srcAccessMask); g.c(':'); g.u(b.dstStageMask); g.c(':'); g.u(b.dstAccessMask);
    g.c(':'); g.u(b.srcQueueFamilyIndex); g.c(':'); g.u(b.dstQueueFamilyIndex); g.c(':'); g.u(b.offset); g.c(':'); g.u(b.size);
  }
  for (uint32_t i = 0; i < di->imageMemoryBarrierCount; i++) {
    auto& x = di->pImageMemoryBarriers[i];
    g.s += "|I:"; g.u(x.srcStageMask); g.c(':'); g.u(x.srcAccessMask); g.c(':'); g.u(x.dstStageMask); g.c(':'); g.u(x.dstAccessMask);
    g.c(':'); g.u(x.srcQueueFamilyIndex); g.c(':'); g.u(x.dstQueueFamilyIndex); g.c(':'); g.u((uint32_t)x.oldLayout);
    g.c(':'); g.u((uint32_t)x.newLayout); g.c(':'); g.u(x.subresourceRange.aspectMask); g.c(':'); g.u(x.subresourceRange.baseMipLevel);
    g.c(':'); g.u(x.subresourceRange.levelCount); g.c(':'); g.u(x.subresourceRange.baseArrayLayer); g.c(':'); g.u(x.subresourceRange.layerCount);
  }
  return g.s;
}

// Advance the command buffer's barrier counter/chain and return the matching
// plan entry (or null). Copies the entry so the lock is held only briefly.
bool lookup(VkCommandBuffer cb, const std::string& sig, Entry& out, uint64_t& rec, uint32_t& idx) {
  Global& g = G();
  g.st.calls++;
  uint64_t h = fnv(sig);
  std::lock_guard<std::mutex> lk(g.mu);
  auto it = g.cbs.find(cb);
  if (it == g.cbs.end()) return false;                 // begun before the layer saw it
  auto& c = it->second;
  rec = c.rec; idx = c.idx++;
  c.chain = fnv(sig, c.chain);
  if (!g.active) return false;
  const Entry* e = nullptr;
  if (g.chain_mode) { auto f = g.chain.find(c.chain); if (f != g.chain.end()) e = &f->second; }
  else { auto f = g.strict.find((rec << 32) | idx); if (f != g.strict.end()) e = &f->second; }
  if (!e) return false;
  if (e->sig != h) { g.st.sig_mismatch++; if (g.verbose) vlog("rec %llu idx %u: signature differs from the plan; forwarded unchanged", (unsigned long long)rec, idx); return false; }
  g.st.matched++;
  out = *e;
  return true;
}

// ------------------------------------------------------------------ hooks
VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* bi) {
  Global& g = G();
  PFN_vkBeginCommandBuffer next;
  {
    std::lock_guard<std::mutex> lk(g.mu);
    auto& c = g.cbs[cb];
    c.rec = g.next_rec++; c.idx = 0; c.chain = FNV_OFFSET;
    next = g.devs[key(cb)].begin;
  }
  g.st.recordings++;
  return next(cb, bi);
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags ss, VkPipelineStageFlags ds, VkDependencyFlags df,
                                              uint32_t nm, const VkMemoryBarrier* m, uint32_t nb, const VkBufferMemoryBarrier* b,
                                              uint32_t ni, const VkImageMemoryBarrier* im) {
  Global& g = G();
  PFN_vkCmdPipelineBarrier next;
  { std::lock_guard<std::mutex> lk(g.mu); next = g.devs[key(cb)].barrier1; }
  Entry e; uint64_t rec; uint32_t idx;
  if (!lookup(cb, sig1(ss, ds, df, nm, m, nb, b, ni, im), e, rec, idx)) { next(cb, ss, ds, df, nm, m, nb, b, ni, im); return; }
  if (e.remove) {
    g.st.removed_calls++; g.st.structs_removed += (nm + nb + ni) ? (nm + nb + ni) : 1;
    if (g.verbose) vlog("rec %llu idx %u: vkCmdPipelineBarrier removed", (unsigned long long)rec, idx);
    return;
  }
  // narrow: sync1 has one stage mask per call
  if (e.has_sync1 && ((e.sync1_ss >> 32) || (e.sync1_ds >> 32))) { next(cb, ss, ds, df, nm, m, nb, b, ni, im); return; }
  std::vector<VkMemoryBarrier> M; std::vector<VkBufferMemoryBarrier> B; std::vector<VkImageMemoryBarrier> I;
  uint32_t removed = 0, narrowed = 0;
  auto apply = [&](char kind, uint32_t i, VkAccessFlags& sa, VkAccessFlags& da) -> bool {
    const ElemPlan* p = e.find(kind, i);
    if (!p || p->action == KEEP) return true;
    if (p->action == REMOVE) { removed++; return false; }
    sa = (VkAccessFlags)p->sa; da = (VkAccessFlags)p->da; narrowed++;
    return true;
  };
  for (uint32_t i = 0; i < nm; i++) { auto x = m[i]; if (apply('M', i, x.srcAccessMask, x.dstAccessMask)) M.push_back(x); }
  for (uint32_t i = 0; i < nb; i++) { auto x = b[i]; if (apply('B', i, x.srcAccessMask, x.dstAccessMask)) B.push_back(x); }
  for (uint32_t i = 0; i < ni; i++) { auto x = im[i]; if (apply('I', i, x.srcAccessMask, x.dstAccessMask)) I.push_back(x); }
  VkPipelineStageFlags nss = e.has_sync1 ? (VkPipelineStageFlags)e.sync1_ss : ss;
  VkPipelineStageFlags nds = e.has_sync1 ? (VkPipelineStageFlags)e.sync1_ds : ds;
  if (!nss) nss = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  if (!nds) nds = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  g.st.narrowed_calls++; g.st.structs_removed += removed; g.st.structs_narrowed += narrowed;
  if (g.verbose) vlog("rec %llu idx %u: vkCmdPipelineBarrier narrowed (stages 0x%x->0x%x, %u structs removed, %u narrowed)",
                      (unsigned long long)rec, idx, nss, nds, removed, narrowed);
  if (M.empty() && B.empty() && I.empty() && (nm + nb + ni) > 0) return;   // nothing left
  next(cb, nss, nds, df, (uint32_t)M.size(), M.data(), (uint32_t)B.size(), B.data(), (uint32_t)I.size(), I.data());
}

void barrier2(VkCommandBuffer cb, const VkDependencyInfo* di, bool khr) {
  Global& g = G();
  PFN_vkCmdPipelineBarrier2 next;
  {
    std::lock_guard<std::mutex> lk(g.mu);
    auto& d = g.devs[key(cb)];
    next = khr && d.barrier2khr ? d.barrier2khr : d.barrier2;
  }
  Entry e; uint64_t rec; uint32_t idx;
  if (!lookup(cb, sig2(di), e, rec, idx)) { next(cb, di); return; }
  uint32_t total = di->memoryBarrierCount + di->bufferMemoryBarrierCount + di->imageMemoryBarrierCount;
  if (e.remove) {
    g.st.removed_calls++; g.st.structs_removed += total;
    if (g.verbose) vlog("rec %llu idx %u: vkCmdPipelineBarrier2 removed", (unsigned long long)rec, idx);
    return;
  }
  std::vector<VkMemoryBarrier2> M; std::vector<VkBufferMemoryBarrier2> B; std::vector<VkImageMemoryBarrier2> I;
  uint32_t removed = 0, narrowed = 0;
  auto apply = [&](char kind, uint32_t i, VkPipelineStageFlags2& ss, VkAccessFlags2& sa, VkPipelineStageFlags2& ds, VkAccessFlags2& da) {
    const ElemPlan* p = e.find(kind, i);
    if (!p || p->action == KEEP) return true;
    if (p->action == REMOVE) { removed++; return false; }
    ss = p->ss; sa = p->sa; ds = p->ds; da = p->da; narrowed++;
    return true;
  };
  for (uint32_t i = 0; i < di->memoryBarrierCount; i++) {
    auto x = di->pMemoryBarriers[i];
    if (apply('M', i, x.srcStageMask, x.srcAccessMask, x.dstStageMask, x.dstAccessMask)) M.push_back(x);
  }
  for (uint32_t i = 0; i < di->bufferMemoryBarrierCount; i++) {
    auto x = di->pBufferMemoryBarriers[i];
    if (apply('B', i, x.srcStageMask, x.srcAccessMask, x.dstStageMask, x.dstAccessMask)) B.push_back(x);
  }
  for (uint32_t i = 0; i < di->imageMemoryBarrierCount; i++) {
    auto x = di->pImageMemoryBarriers[i];
    if (apply('I', i, x.srcStageMask, x.srcAccessMask, x.dstStageMask, x.dstAccessMask)) I.push_back(x);
  }
  g.st.narrowed_calls++; g.st.structs_removed += removed; g.st.structs_narrowed += narrowed;
  if (g.verbose) vlog("rec %llu idx %u: vkCmdPipelineBarrier2 narrowed (%u structs removed, %u narrowed)", (unsigned long long)rec, idx, removed, narrowed);
  if (M.empty() && B.empty() && I.empty()) return;
  VkDependencyInfo d2 = *di;
  d2.memoryBarrierCount = (uint32_t)M.size(); d2.pMemoryBarriers = M.data();
  d2.bufferMemoryBarrierCount = (uint32_t)B.size(); d2.pBufferMemoryBarriers = B.data();
  d2.imageMemoryBarrierCount = (uint32_t)I.size(); d2.pImageMemoryBarriers = I.data();
  next(cb, &d2);
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo* di) { barrier2(cb, di, false); }
VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier2KHR(VkCommandBuffer cb, const VkDependencyInfo* di) { barrier2(cb, di, true); }

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice dev, const VkAllocationCallbacks* a) {
  Global& g = G();
  PFN_vkDestroyDevice next;
  {
    std::lock_guard<std::mutex> lk(g.mu);
    auto it = g.devs.find(key(dev));
    next = it->second.destroy;
    g.devs.erase(it);
  }
  next(dev, a);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo* ci, const VkAllocationCallbacks* a, VkDevice* out) {
  auto* info = (VkLayerDeviceCreateInfo*)ci->pNext;
  while (info && !(info->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && info->function == VK_LAYER_LINK_INFO))
    info = (VkLayerDeviceCreateInfo*)info->pNext;
  if (!info) return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkGetInstanceProcAddr gipa = info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr gdpa = info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
  info->u.pLayerInfo = info->u.pLayerInfo->pNext;
  auto create = (PFN_vkCreateDevice)gipa(VK_NULL_HANDLE, "vkCreateDevice");
  VkResult r = create(phys, ci, a, out);
  if (r != VK_SUCCESS) return r;
  Global::Dev d;
  d.gdpa = gdpa;
  d.destroy = (PFN_vkDestroyDevice)gdpa(*out, "vkDestroyDevice");
  d.begin = (PFN_vkBeginCommandBuffer)gdpa(*out, "vkBeginCommandBuffer");
  d.barrier1 = (PFN_vkCmdPipelineBarrier)gdpa(*out, "vkCmdPipelineBarrier");
  d.barrier2 = (PFN_vkCmdPipelineBarrier2)gdpa(*out, "vkCmdPipelineBarrier2");
  d.barrier2khr = (PFN_vkCmdPipelineBarrier2)gdpa(*out, "vkCmdPipelineBarrier2KHR");
  std::lock_guard<std::mutex> lk(G().mu);
  G().devs[key(*out)] = d;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo* ci, const VkAllocationCallbacks* a, VkInstance* out) {
  load_plan();
  auto* info = (VkLayerInstanceCreateInfo*)ci->pNext;
  while (info && !(info->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && info->function == VK_LAYER_LINK_INFO))
    info = (VkLayerInstanceCreateInfo*)info->pNext;
  if (!info) return VK_ERROR_INITIALIZATION_FAILED;
  PFN_vkGetInstanceProcAddr gipa = info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  info->u.pLayerInfo = info->u.pLayerInfo->pNext;
  auto create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
  VkResult r = create(ci, a, out);
  if (r != VK_SUCCESS) return r;
  std::lock_guard<std::mutex> lk(G().mu);
  G().inst_gipa[key(*out)] = gipa;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance inst, const VkAllocationCallbacks* a) {
  PFN_vkGetInstanceProcAddr gipa;
  {
    std::lock_guard<std::mutex> lk(G().mu);
    gipa = G().inst_gipa[key(inst)];
    G().inst_gipa.erase(key(inst));
  }
  auto destroy = (PFN_vkDestroyInstance)gipa(inst, "vkDestroyInstance");
  destroy(inst, a);
}

PFN_vkVoidFunction device_hook(const char* n) {
  if (!strcmp(n, "vkBeginCommandBuffer")) return (PFN_vkVoidFunction)BeginCommandBuffer;
  if (!strcmp(n, "vkCmdPipelineBarrier")) return (PFN_vkVoidFunction)CmdPipelineBarrier;
  if (!strcmp(n, "vkCmdPipelineBarrier2")) return (PFN_vkVoidFunction)CmdPipelineBarrier2;
  if (!strcmp(n, "vkCmdPipelineBarrier2KHR")) return (PFN_vkVoidFunction)CmdPipelineBarrier2KHR;
  if (!strcmp(n, "vkDestroyDevice")) return (PFN_vkVoidFunction)DestroyDevice;
  return nullptr;
}

// Internal entry points. The exported vk* symbols below only forward to these:
// inside a process that also loaded libvulkan, a reference to an exported
// "vkGetInstanceProcAddr" could otherwise bind to the loader's own symbol.
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice dev, const char* n);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance inst, const char* n) {
  if (!strcmp(n, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)GetInstanceProcAddr;
  if (!strcmp(n, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)GetDeviceProcAddr;
  if (!strcmp(n, "vkCreateInstance")) return (PFN_vkVoidFunction)CreateInstance;
  if (!strcmp(n, "vkDestroyInstance")) return (PFN_vkVoidFunction)DestroyInstance;
  if (!strcmp(n, "vkCreateDevice")) return (PFN_vkVoidFunction)CreateDevice;
  if (auto f = device_hook(n)) return f;
  if (!inst) return nullptr;
  PFN_vkGetInstanceProcAddr gipa;
  {
    std::lock_guard<std::mutex> lk(G().mu);
    auto it = G().inst_gipa.find(key(inst));
    if (it == G().inst_gipa.end()) return nullptr;
    gipa = it->second;
  }
  return gipa(inst, n);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice dev, const char* n) {
  if (!strcmp(n, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)GetDeviceProcAddr;
  if (auto f = device_hook(n)) return f;
  PFN_vkGetDeviceProcAddr gdpa;
  {
    std::lock_guard<std::mutex> lk(G().mu);
    auto it = G().devs.find(key(dev));
    if (it == G().devs.end()) return nullptr;
    gdpa = it->second.gdpa;
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
