// Distilled from pixijs/pixijs PR #12154 ("fix: preserve UBO offsets in
// unsafe-eval polyfill", fixes #12106, merged 2026-08-24). Under a strict CSP
// PixiJS cannot generate UBO upload code with new Function(), so
// pixi.js/unsafe-eval installs a polyfill (generateUboSyncPolyfill). The UBO
// sync contract is
//   syncFunction(uniforms, data: Float32Array, dataInt32: Int32Array | null, offset)
// and UboSystem.syncUniformGroup() calls it with all four arguments. The
// polyfill returned a three-parameter function (uniforms, data, offset), so
// it took `offset` from the dataInt32 argument -- null on the WebGPU batch
// path, which coerces to 0. GpuUniformBatchPipe packs every uniform group of
// a frame into one shared buffer at 256-byte-aligned offsets and binds each
// group at its own offset, but the polyfill wrote every group's values at
// offset 0: the groups overwrote each other there and the meshes bound at
// 256 and 512 read zeros. The fix adds the missing `_dataInt32` parameter.
// Port: TypeScript -> C++; the polyfill's positional argument mix-up is kept
// as a callback whose third parameter is used as the offset (with JS's
// null -> 0 coercion written out); WebGPU bind groups with buffer offsets ->
// descriptor sets whose VkDescriptorBufferInfo.offset is the group's offset.
#include "../../common/app.hpp"
#include <functional>
using namespace bench;

constexpr uint32_t minUniformOffsetAlignment = 256;

struct Mat3Std140 { float c[3][4]; };
static Mat3Std140 mat3(float a, float b, float c, float d, float tx, float ty) {
  Mat3Std140 m{}; m.c[0][0] = a; m.c[0][1] = b; m.c[1][0] = c; m.c[1][1] = d; m.c[2][0] = tx; m.c[2][1] = ty; m.c[2][2] = 1;
  return m;
}
struct UniformGroup { Mat3Std140 uTransformMatrix; float uColor[4]; float uRound; };   // localUniforms
struct UboElement { const char* name; uint32_t offset; uint32_t size; };
static const UboElement kLocalLayout[3] = {{"uTransformMatrix", 0, 48}, {"uColor", 48, 16}, {"uRound", 64, 4}};
constexpr uint32_t kLocalLayoutSize = 80;   // uboElements layout size (16-byte rounded)

// UboSystem's sync function contract
using UboSyncFunction = std::function<void(const UniformGroup& uniforms, float* data, const int32_t* dataInt32, uint32_t offset)>;

// pixi.js/unsafe-eval: generateUboSyncPolyfill(uboElements)
static UboSyncFunction generateUboSyncPolyfill(const UboElement* elements, int count) {
  std::vector<UboElement> functionMap(elements, elements + count);
#ifdef FIXED
  return [functionMap](const UniformGroup& uniforms, float* data, const int32_t* /*_dataInt32*/, uint32_t offset) {
#else
  // (uniforms, data, offset) => ...: the third argument is taken as the offset
  return [functionMap](const UniformGroup& uniforms, float* data, const int32_t* offsetArg, uint32_t) {
    const uint32_t offset = offsetArg ? (uint32_t)(uintptr_t)offsetArg : 0;   // JS: null + n === n
#endif
    for (const auto& e : functionMap) {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(&uniforms) + e.offset;   // uniforms[name]
      memcpy(data + offset + e.offset / 4, src, e.size);
    }
  };
}

// GpuUniformBatchPipe's shared buffer
struct UniformBufferBatch {
  std::vector<float> data = std::vector<float>(4096 / 4, 0.f);
  uint32_t byteIndex = 0;
  uint32_t addEmptyGroup(uint32_t sizeFloats) {
    uint32_t start = byteIndex;
    byteIndex += sizeFloats * 4;
    byteIndex = (byteIndex + minUniformOffsetAlignment - 1) / minUniformOffsetAlignment * minUniformOffsetAlignment;
    return start;
  }
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  const float W = (float)a.extent.width, H = (float)a.extent.height;

  struct GlobalUniforms { Mat3Std140 uProjectionMatrix; Mat3Std140 uWorldTransformMatrix; float uWorldColorAlpha[4]; float uResolution[2]; float _pad[2]; } g{};
  g.uProjectionMatrix = mat3(2.f / W, 0, 0, 2.f / H, -1, -1);
  g.uWorldTransformMatrix = mat3(1, 0, 0, 1, 0, 0);
  g.uWorldColorAlpha[0] = g.uWorldColorAlpha[1] = g.uWorldColorAlpha[2] = g.uWorldColorAlpha[3] = 1;
  g.uResolution[0] = W; g.uResolution[1] = H;
  Buffer globalBuf = make_buffer(a, sizeof g, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "globalUniforms");
  memcpy(globalBuf.map, &g, sizeof g);

  // three meshes, each with its own local uniform group
  UniformGroup groups[3] = {{mat3(1, 0, 0, 1, 60, 60), {1, 0.2f, 0.2f, 1}, 0},
                            {mat3(1, 0, 0, 1, 240, 60), {0.2f, 1, 0.2f, 1}, 0},
                            {mat3(1, 0, 0, 1, 420, 60), {0.2f, 0.4f, 1, 1}, 0}};
  UboSyncFunction syncFunction = generateUboSyncPolyfill(kLocalLayout, 3);

  Buffer batchBuffer = make_buffer(a, 4096, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "uniform batch buffer[0]");
  auto set0l = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto set1l = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT}});
  auto globalSet = alloc_set(a, set0l, "globalUniformBindGroup");
  write_buffer(a, globalSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, globalBuf);

  // 128x128 quad
  const float quad[12] = {0, 0, 128, 0, 0, 128, 0, 128, 128, 0, 128, 128};
  Buffer vb = make_buffer(a, sizeof quad, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "aPosition");
  memcpy(vb.map, quad, sizeof quad);
  GfxDesc gd; gd.vs = gd.fs = sh; gd.vs_entry = "mainVertex"; gd.fs_entry = "mainFragment";
  gd.sets = {set0l, set1l}; gd.colors = {a.swap_format}; gd.vertex_stride = 8;
  gd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}};
  Pipeline mesh = make_graphics(a, gd, "mesh pipeline");

  // one bind group per batch-buffer slot (GpuUniformBatchPipe._getBindGroup(index))
  std::vector<VkDescriptorSet> bindGroups;
  for (int i = 0; i < 3; i++) {
    VkDescriptorSet s = alloc_set(a, set1l, "uniformBatchBindGroup[" + std::to_string(i) + "]");
    VkDescriptorBufferInfo bi{batchBuffer.buf, (VkDeviceSize)i * minUniformOffsetAlignment, kLocalLayoutSize};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = s; w.dstBinding = 0; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(a.dev, 1, &w, 0, nullptr);
    bindGroups.push_back(s);
  }

  run_frames(a, frames_from_env(2), 1, [&](VkCommandBuffer c, uint32_t idx, int) {
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh.layout, 0, 1, &globalSet, 0, nullptr);
    VkDeviceSize off = 0; vkCmdBindVertexBuffers(c, 0, 1, &vb.buf, &off);
    for (int i = 0; i < 3; i++) {
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh.layout, 1, 1, &bindGroups[i], 0, nullptr);
      vkCmdDraw(c, 6, 1, 0, 0);
    }
    vkCmdEndRendering(c);
  }, [&](int) {
    // renderStart: getUniformBindGroup(group) for each mesh, then renderEnd uploads the batch
    UniformBufferBatch batch;
    for (int i = 0; i < 3; i++) {
      uint32_t offset = batch.addEmptyGroup(kLocalLayoutSize / 4);
      // UboSystem.syncUniformGroup(group, batch.data, offset / 4): dataInt32 stays null
      const int32_t* dataInt32 = nullptr;
      syncFunction(groups[i], batch.data.data(), dataInt32, offset / 4);
    }
    memcpy(batchBuffer.map, batch.data.data(), batch.byteIndex);   // _uploadBindGroups
  });
  return 0;
}
