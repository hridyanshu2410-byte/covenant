// Present-limit shim for the validation runs (harness only, no app or driver change): ends the process
// right after the COV_PRESENTS-th successful vkQueuePresentKHR (default 20), through exit(0) so atexit
// handlers run. Works for apps that load Vulkan with dlopen("libvulkan.so.1") + dlsym("vkGetInstanceProcAddr")
// (SDL3): dlsym on that handle returns a wrapper whose vkGetInstanceProcAddr / vkGetDeviceProcAddr hand
// out a counting vkQueuePresentKHR. Lookups on any other handle (the loader's own dlsym of layers/ICDs)
// are untouched. It also interposes vkQueuePresentKHR directly for programs that link libvulkan and call
// it through the PLT (Dear ImGui). Build: cc -O2 -shared -fPIC -o presentlimit.so presentlimit.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VISIBLE __attribute__((visibility("default")))
typedef void (*PFN)(void);
typedef PFN (*GIPA)(void*, const char*);
typedef int (*PRESENT)(void*, const void*);
static GIPA real_gipa, real_gdpa;
static PRESENT real_present;
static int presents;
static void* (*real_dlsym)(void*, const char*);
static int limit(void) { const char* s = getenv("COV_PRESENTS"); return s ? atoi(s) : 20; }
static int my_present(void* q, const void* info) {
  int r = real_present(q, info);
  if (r >= 0 && ++presents >= limit()) {
    fprintf(stderr, "[presentlimit] %d presents, exiting\n", presents);
    exit(0);
  }
  return r;
}
static PFN my_gdpa(void* dev, const char* name) {
  PFN f = real_gdpa(dev, name);
  if (f && !strcmp(name, "vkQueuePresentKHR")) { real_present = (PRESENT)f; return (PFN)my_present; }
  return f;
}
static PFN my_gipa(void* inst, const char* name) {
  PFN f = real_gipa(inst, name);
  if (!f) return f;
  if (!strcmp(name, "vkQueuePresentKHR")) { real_present = (PRESENT)f; return (PFN)my_present; }
  if (!strcmp(name, "vkGetDeviceProcAddr")) { real_gdpa = (GIPA)f; return (PFN)my_gdpa; }
  if (!strcmp(name, "vkGetInstanceProcAddr")) return (PFN)my_gipa;
  return f;
}
void* dlsym(void* handle, const char* name) {
  if (!real_dlsym) real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
  if (!real_dlsym) real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
  void* r = real_dlsym(handle, name);
  if (r && handle != RTLD_NEXT && handle != RTLD_DEFAULT) {
    void* vk = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (vk) dlclose(vk);
    if (vk == handle) {
      if (!strcmp(name, "vkGetInstanceProcAddr")) { real_gipa = (GIPA)r; return (void*)my_gipa; }
      if (!strcmp(name, "vkGetDeviceProcAddr")) { real_gdpa = (GIPA)r; return (void*)my_gdpa; }
      if (!strcmp(name, "vkQueuePresentKHR")) { real_present = (PRESENT)r; return (void*)my_present; }
    }
  }
  return r;
}
VISIBLE int vkQueuePresentKHR(void* q, const void* info) {
  if (!real_present) {
    void* (*ds)(void*, const char*) = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (!ds) ds = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    real_present = (PRESENT)ds(RTLD_NEXT, "vkQueuePresentKHR");
  }
  return my_present(q, info);
}
__attribute__((destructor)) static void report(void) { fprintf(stderr, "[presentlimit] presents=%d\n", presents); }
