// Host-write capture probes for the capture modes (tests/test_strict_undefined_and_capture.py).
//
//   hostcapture <mode>
//
// One host-visible, coherent memory object holds two 256-byte buffers in one page:
// `src` (bytes 0..255) and `nbr` (bytes 256..511). A copy reads `src` into a device
// buffer and is submitted with a fence. Then, depending on <mode>, the host stores to
// the mapping BEFORE waiting for the fence (so the store races the copy's read), and
// finally waits for the fence:
//   ok        no store before the wait; one changed store after it (no race)
//   changed   stores a different value into src (a race; every capture mode sees it)
//   same      stores the value src already holds (a race; no byte changes, so the
//             observed-modification capture does not see it)
//   restore   stores a different value and then the original one back (a race; the
//             bytes are unchanged at the next Vulkan call)
//   neighbour stores into nbr only, which shares src's page (no race: the conservative
//             capture charges the whole page and reports one, the page-granularity cost)
//   thread    like `changed`, but the store is made by a second thread that is joined
//             before the fence wait (flagged: the capture assumes one writing thread)
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define CHECK(x)                                                              \
  do {                                                                        \
    VkResult r_ = (x);                                                        \
    if (r_ != VK_SUCCESS) {                                                   \
      fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, r_); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static uint32_t find_memory(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  fprintf(stderr, "no suitable memory type\n");
  exit(1);
}

static VkBuffer make_buffer(VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage) {
  VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage;
  VkBuffer b;
  CHECK(vkCreateBuffer(dev, &bi, nullptr, &b));
  return b;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "ok";
  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  CHECK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 1;
  VkPhysicalDevice pd;
  vkEnumeratePhysicalDevices(inst, &n, &pd);
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = 0;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice dev;
  CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, 0, 0, &q);

  // two 256-byte buffers in the first page of one host-visible allocation, one device buffer
  VkBuffer src = make_buffer(dev, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer nbr = make_buffer(dev, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer dst = make_buffer(dev, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, src, &mr);
  VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = 4096;
  mai.memoryTypeIndex = find_memory(pd, mr.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkDeviceMemory hmem, dmem;
  CHECK(vkAllocateMemory(dev, &mai, nullptr, &hmem));
  mai.memoryTypeIndex = find_memory(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  CHECK(vkAllocateMemory(dev, &mai, nullptr, &dmem));
  CHECK(vkBindBufferMemory(dev, src, hmem, 0));
  CHECK(vkBindBufferMemory(dev, nbr, hmem, 256));
  CHECK(vkBindBufferMemory(dev, dst, dmem, 0));
  unsigned char* p = nullptr;
  CHECK(vkMapMemory(dev, hmem, 0, VK_WHOLE_SIZE, 0, (void**)&p));
  memset(p, 0x11, 256);                 // the data the copy reads
  memset(p + 256, 0x22, 256);

  VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.queueFamilyIndex = 0;
  VkCommandPool pool;
  CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
  VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  VkCommandBuffer cb;
  CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
  VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  CHECK(vkBeginCommandBuffer(cb, &cbi));
  VkBufferCopy region = {0, 0, 256};
  vkCmdCopyBuffer(cb, src, dst, 1, &region);
  CHECK(vkEndCommandBuffer(cb));
  VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  CHECK(vkCreateFence(dev, &fci, nullptr, &fence));
  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  CHECK(vkQueueSubmit(q, 1, &si, fence));

  // ---- the store under test: after the submit, before the fence wait
  volatile unsigned char* v = p;
  if (!strcmp(mode, "changed")) {
    v[16] = 0x33;
  } else if (!strcmp(mode, "same")) {
    v[16] = 0x11;                       // the byte already holds 0x11
  } else if (!strcmp(mode, "restore")) {
    v[16] = 0x44;
    v[16] = 0x11;
  } else if (!strcmp(mode, "neighbour")) {
    v[256 + 16] = 0x55;
  } else if (!strcmp(mode, "thread")) {
    std::thread t([v] { v[16] = 0x66; });
    t.join();
  } else if (strcmp(mode, "ok")) {
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
  }
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
  v[32] = 0x77;                         // after the wait: ordered after the copy
  CHECK(vkDeviceWaitIdle(dev));
  vkDestroyFence(dev, fence, nullptr);
  vkDestroyCommandPool(dev, pool, nullptr);
  vkDestroyDevice(dev, nullptr);
  vkDestroyInstance(inst, nullptr);
  return 0;
}
