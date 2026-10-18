// LD_PRELOAD shim (Linux only): makes std::chrono::steady_clock / system_clock /
// high_resolution_clock advance by exactly 1/60 s per presented frame, so that
// time-based animation in unmodified samples (SaschaWillems/Vulkan computes
// frameTimer and timer from std::chrono) is identical from run to run, and the
// presented images can be compared by hash. VK_LAYER_COVENANT_probe calls
// covenant_fakeclock_tick() at every vkQueuePresentKHR. Only std::chrono is
// faked: the driver, the loader and the validation layer use clock_gettime
// directly and keep the real clock (their timeouts stay real).
//
//   g++ -O2 -shared -fPIC -o libcovenant_fakeclock.so fakeclock.cpp
//   LD_PRELOAD=$PWD/libcovenant_fakeclock.so <sample> ...
#include <atomic>
#include <cstdint>
#include <ctime>

static std::atomic<int64_t> g_frames{0};
static int64_t base_ns(clockid_t id) {
  timespec ts;
  clock_gettime(id, &ts);
  return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}
static const int64_t g_steady0 = base_ns(CLOCK_MONOTONIC);
static const int64_t g_system0 = base_ns(CLOCK_REALTIME);
static const int64_t FRAME_NS = 16666667;   // 1/60 s

extern "C" __attribute__((visibility("default"))) void covenant_fakeclock_tick() { g_frames++; }

// std::chrono::_V2::steady_clock::now() and system_clock::now() (libstdc++ ABI):
// a time_point is one int64 of nanoseconds, returned in a register.
extern "C" __attribute__((visibility("default"))) int64_t covenant_fake_steady_now() asm("_ZNSt6chrono3_V212steady_clock3nowEv");
extern "C" __attribute__((visibility("default"))) int64_t covenant_fake_system_now() asm("_ZNSt6chrono3_V212system_clock3nowEv");
int64_t covenant_fake_steady_now() { return g_steady0 + g_frames.load() * FRAME_NS; }
int64_t covenant_fake_system_now() { return g_system0 + g_frames.load() * FRAME_NS; }
