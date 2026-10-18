// Distilled from amodhakal/linterra #127 (fix PR #182): chunks are culled on
// the CPU against a Frustum built from the camera, and the survivors are
// drawn. Camera::getProjection() builds the projection with
// glm::radians(m_Fov), but Frustum::Frustum computed the side planes with
// camera->m_Far * tan(camera->m_Fov * 0.5), m_Fov being in degrees. The
// culling cone and the rendered cone disagree: at the default 45 degrees
// chunks are kept that are not visible and visible ones dropped, and at
// 25-35 degrees tan goes negative, the side normals flip and the world
// disappears. The fix converts with glm::radians(m_Fov) in the frustum too.
#include "../../common/app.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace bench;

struct Camera {
  glm::vec3 m_Position{0.0f, 24.0f, 0.0f};
  glm::vec3 m_Front = glm::normalize(glm::vec3(1.0f, -0.35f, 1.0f));
  glm::vec3 m_Up{0.0f, 1.0f, 0.0f};
  float m_Fov COV("unit=deg") = 45.0f;
  float m_Aspect = 640.0f / 480.0f;
  float m_Near = 0.1f, m_Far = 256.0f;

  glm::vec3 right() const { return glm::normalize(glm::cross(m_Front, m_Up)); }
  glm::mat4 getView() const { return glm::lookAt(m_Position, m_Position + m_Front, m_Up); }
  glm::mat4 getProjection() const { return glm::perspective(glm::radians(m_Fov), m_Aspect, m_Near, m_Far); }
};

struct Plane { glm::vec3 normal; float distance; };
static Plane plane(glm::vec3 p, glm::vec3 n) { n = glm::normalize(n); return {n, glm::dot(n, p)}; }

struct Frustum {
  Plane top, bottom, right, left, farFace, nearFace;
  explicit Frustum(const Camera* camera) {
#ifdef FIXED
    const float halfVSide = camera->m_Far * std::tan(glm::radians(camera->m_Fov) * 0.5f);
#else
    const float halfVSide = camera->m_Far * std::tan(camera->m_Fov * 0.5f);
#endif
    const float halfHSide = halfVSide * camera->m_Aspect;
    const glm::vec3 frontMultFar = camera->m_Far * camera->m_Front;
    const glm::vec3 r = camera->right(), up = glm::normalize(glm::cross(r, camera->m_Front));
    const glm::vec3 pos = camera->m_Position;
    nearFace = plane(pos + camera->m_Near * camera->m_Front, camera->m_Front);
    farFace = plane(pos + frontMultFar, -camera->m_Front);
    right = plane(pos, glm::cross(frontMultFar - r * halfHSide, up));
    left = plane(pos, glm::cross(up, frontMultFar + r * halfHSide));
    top = plane(pos, glm::cross(r, frontMultFar - up * halfVSide));
    bottom = plane(pos, glm::cross(frontMultFar + up * halfVSide, r));
  }
  static bool outside(const Plane& p, glm::vec3 mn, glm::vec3 mx) {
    glm::vec3 v(p.normal.x >= 0 ? mx.x : mn.x, p.normal.y >= 0 ? mx.y : mn.y, p.normal.z >= 0 ? mx.z : mn.z);
    float distance = glm::dot(p.normal, v) - p.distance;
    return distance < 0;
  }
  bool isChunkInside(glm::vec3 mn, glm::vec3 mx) const {
    for (const Plane* p : {&top, &bottom, &right, &left, &farFace, &nearFace})
      if (outside(*p, mn, mx)) return false;
    return true;
  }
};

struct CameraUniforms { glm::mat4 view; glm::mat4 projection; };
static const int CHUNKS = 32, CHUNK_SIZE = 16;

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  Camera camera;
  Buffer ubo = make_buffer(a, sizeof(CameraUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera");
  Buffer instances = make_buffer(a, CHUNKS * CHUNKS * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true,
                                 "visible_chunks");
  Image depth = make_image(a, 640, 480, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "depth");
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
                                  {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto set = alloc_set(a, setl, "camera_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubo);
  write_buffer(a, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, instances);
  auto sh = load_shader(a, "shaders.spv");
  GfxDesc gd; gd.vs = gd.fs = sh; gd.sets = {setl}; gd.colors = {a.swap_format}; gd.depth = VK_FORMAT_D32_SFLOAT;
  Pipeline terrain = make_graphics(a, gd, "chunks");

  uint32_t visible = 0;
  run_frames(a, frames_from_env(3), 1, [&](VkCommandBuffer c, uint32_t idx, int f) {
    depth_barrier(c, depth.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    begin_rendering(a, c, {a.swap_views[idx]}, depth.view, 1.0f);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain.layout, 0, 1, &set, 0, nullptr);
    if (visible) vkCmdDraw(c, 36, visible, 0, 0);       // one instance per kept chunk
    vkCmdEndRendering(c);
  }, [&](int f) {
    camera.m_Position.x = 4.0f * f;
    CameraUniforms u{camera.getView(), camera.getProjection()};
    u.projection[1][1] *= -1;
    memcpy(ubo.map, &u, sizeof u);
    Frustum frustum(&camera);
    glm::vec4* out = (glm::vec4*)instances.map;
    visible = 0;
    for (int x = 0; x < CHUNKS; x++)
      for (int z = 0; z < CHUNKS; z++) {
        glm::vec3 mn(x * CHUNK_SIZE, 0, z * CHUNK_SIZE), mx = mn + glm::vec3(CHUNK_SIZE, 16, CHUNK_SIZE);
        if (frustum.isChunkInside(mn, mx)) out[visible++] = glm::vec4(mn, 1);
      }
  });
  printf("visible chunks: %u\n", visible);
}
