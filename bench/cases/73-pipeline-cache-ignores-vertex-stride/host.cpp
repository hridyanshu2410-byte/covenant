// Distilled from mrdoob/three.js #34722 (fixed by PR #34726, r187):
// WebGPURenderer identifies render pipelines by a render cache key that
// contained renderObject.getGeometryCacheKey(). A StorageBufferAttribute
// position (itemSize 3, padded to 4 floats: vertex stride 16) and an
// ordinary BufferAttribute position (stride 12) produce the same geometry
// key, so after swapping an InstancedMesh's geometry from a storage-attribute
// geometry to a plain one, the pipeline built for stride 16 was reused with
// the 72-byte, 12-byte-stride buffer: "Vertex range (first: 0, count: 6)
// requires a larger buffer (96) than the bound buffer size (72) of the vertex
// buffer at slot 0 with stride 16." The fix removes getGeometryCacheKey()
// from the pipeline key and adds a vertex buffer layout key (computed in
// WebGPUAttributeUtils) and the strip index format. (Upstream needed a
// storage -> storage -> plain sequence to hit the stale entry; here one swap
// does, since the key below never includes the layout.)
#include "../../common/app.hpp"
#include <map>
using namespace bench;

struct Geometry { const char* name; Buffer position; uint32_t stride; uint32_t count; };

static std::string geometry_cache_key(const Geometry& g) { (void)g; return "position"; }   // attribute names only
static std::string vertex_buffer_layout_key(const Geometry& g) { return "float32x3:" + std::to_string(g.stride); }

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  Buffer cam = make_buffer(a, 64, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera");
  float id[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.5f, 1}; memcpy(cam.map, id, sizeof id);
  Buffer inst = make_buffer(a, 4 * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "instanceMatrix");
  float offs[16] = {-0.5f, -0.5f, 0, 0, 0.5f, -0.5f, 0, 0, -0.5f, 0.5f, 0, 0, 0.5f, 0.5f, 0, 0}; memcpy(inst.map, offs, sizeof offs);
  auto set = alloc_set(a, setl, "renderObject_bindings");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cam);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, inst);

  const float quad[6][3] = {{-0.2f, -0.2f, 0}, {0.2f, -0.2f, 0}, {-0.2f, 0.2f, 0}, {0.2f, -0.2f, 0}, {0.2f, 0.2f, 0}, {-0.2f, 0.2f, 0}};
  // geometryA: StorageBufferAttribute (vec3 padded to vec4) -> stride 16, 96 bytes
  Geometry storageGeo{"storage_geometry", make_buffer(a, 96, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true,
                                                      "StorageBufferAttribute.position"), 16, 6};
  // geometryB: BufferAttribute (tightly packed vec3) -> stride 12, 72 bytes
  Geometry plainGeo{"plain_geometry", make_buffer(a, 72, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "BufferAttribute.position"), 12, 6};
  for (int v = 0; v < 6; v++) {
    memcpy((char*)storageGeo.position.map + v * 16, quad[v], 12);
    memcpy((char*)plainGeo.position.map + v * 12, quad[v], 12);
  }

  std::map<std::string, Pipeline> pipelines;           // Pipelines.getForRender()
  auto pipeline_for = [&](const Geometry& g) -> Pipeline& {
#ifdef FIXED
    std::string key = "MeshStandardNodeMaterial|" + vertex_buffer_layout_key(g) + "|triangle-list";
#else
    std::string key = "MeshStandardNodeMaterial|" + geometry_cache_key(g) + "|triangle-list";
#endif
    auto it = pipelines.find(key);
    if (it != pipelines.end()) return it->second;
    GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format};
    gd.vertex_stride = g.stride; gd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
    return pipelines[key] = make_graphics(a, gd, std::string("render_pipeline<") + g.name + ">");
  };

  run_frames(a, frames_from_env(4), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    const Geometry& g = f < 2 ? storageGeo : plainGeo;   // mesh.geometry = geometryB after two frames
    Pipeline& p = pipeline_for(g);
    begin_rendering(a, c, {a.swap_views[idx]});
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(c, 0, 1, &g.position.buf, &off);
    vkCmdDraw(c, g.count, 4, 0, 0);
    vkCmdEndRendering(c);
  });
}
