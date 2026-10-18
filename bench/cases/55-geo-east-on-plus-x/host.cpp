// Distilled from openzigs/onyourleft PR #645: the cycling game projects
// route coordinates (longitude, latitude, elevation) into the three.js
// world in terrain.ts localGroundPosition, with east on +x, north on +z and
// +y up. three.js is right-handed, so with north on +z the east direction is
// -x: a camera behind a northbound rider has -x (west) on its right, and the
// whole world renders as a mirror image of the map; a right-hand bend on the
// north-up HUD plan turns left on screen. The fix negates the east
// component, x = (origin.longitude - longitude) * metresPerDegreeLongitude;
// everything built from the corridor by rotations needs no change.
// (TypeScript/three.js upstream; the projection is host C++ with glm here,
// and the HUD plan is drawn from the same route.)
#include "../../common/app.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace bench;

struct LatLon { double latitude, longitude; float elevation; };
static const double kMetresPerDegreeLatitude = 111320.0;

// terrain.ts: localGroundPosition(point, origin)
static glm::vec3 local_ground_position(const LatLon& p, const LatLon& origin) {
  double metres_per_degree_longitude = kMetresPerDegreeLatitude * std::cos(origin.latitude * M_PI / 180.0);
#ifdef FIXED
  float x = float((origin.longitude - p.longitude) * metres_per_degree_longitude);
#else
  float x = float((p.longitude - origin.longitude) * metres_per_degree_longitude);    // east on +x
#endif
  float z = float((p.latitude - origin.latitude) * kMetresPerDegreeLatitude);         // north on +z
  return {x, p.elevation, z};
}

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  auto sh = load_shader(a, "shaders.spv");
  // a route heading north, then bending right (east)
  const LatLon origin{51.5000, -0.1200, 10.0f};
  std::vector<LatLon> route;
  for (int i = 0; i < 8; i++) route.push_back({51.5000 + i * 0.0002, -0.1200, 10.0f});
  for (int i = 1; i < 8; i++) route.push_back({51.5014 + i * 0.0001, -0.1200 + i * 0.0003, 10.0f});

  // 3D road ribbon: two vertices per route point, 6 m wide, built with rotations of the corridor
  std::vector<glm::vec3> world;
  for (size_t i = 0; i < route.size(); i++) world.push_back(local_ground_position(route[i], origin));
  std::vector<float> ribbon;
  for (size_t i = 0; i + 1 < world.size(); i++) {
    glm::vec3 d = glm::normalize(world[i + 1] - world[i]);
    glm::vec3 side = glm::cross(d, glm::vec3(0, 1, 0)) * 3.0f;
    glm::vec3 q[6] = {world[i] - side, world[i] + side, world[i + 1] - side,
                      world[i] + side, world[i + 1] + side, world[i + 1] - side};
    for (auto& v : q) { ribbon.push_back(v.x); ribbon.push_back(v.y); ribbon.push_back(v.z); }
  }
  Buffer road = make_buffer(a, ribbon.size() * 4, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "road_ribbon");
  memcpy(road.map, ribbon.data(), ribbon.size() * 4);

  // HUD plan: north up, east right (screen space, from lat/lon directly)
  std::vector<float> plan;
  auto plan_xy = [&](const LatLon& p) {
    return glm::vec2(float((p.longitude - origin.longitude) * 100.0) + 0.6f,
                     float((p.latitude - origin.latitude) * 100.0) - 0.9f);
  };
  for (size_t i = 0; i + 1 < route.size(); i++) {
    glm::vec2 p0 = plan_xy(route[i]), p1 = plan_xy(route[i + 1]);
    glm::vec2 d = glm::normalize(p1 - p0), n(-d.y * 0.006f, d.x * 0.006f);
    glm::vec2 q[6] = {p0 - n, p0 + n, p1 - n, p0 + n, p1 + n, p1 - n};
    for (auto& v : q) { plan.push_back(v.x); plan.push_back(v.y); }
  }
  Buffer hud = make_buffer(a, plan.size() * 4, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, "hud_plan");
  memcpy(hud.map, plan.data(), plan.size() * 4);

  // chase camera behind the rider (heading north)
  glm::vec3 rider = world[3];
  glm::vec3 heading = glm::normalize(world[4] - world[3]);
  glm::mat4 view = glm::lookAt(rider - heading * 12.0f + glm::vec3(0, 4, 0), rider + heading * 20.0f, glm::vec3(0, 1, 0));
  glm::mat4 proj = glm::perspective(glm::radians(60.0f), 640.0f / 480.0f, 0.1f, 2000.0f);
  proj[1][1] *= -1;   // Vulkan clip y
  glm::mat4 view_proj = proj * view;
  Buffer camera = make_buffer(a, sizeof(glm::mat4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "camera");
  memcpy(camera.map, &view_proj, sizeof view_proj);
  auto setl = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}});
  auto set = alloc_set(a, setl, "camera_set");
  write_buffer(a, set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, camera);

  GfxDesc rd; rd.vs = rd.fs = sh; rd.vs_entry = "vsRoad"; rd.fs_entry = "fsRoad"; rd.sets = {setl};
  rd.colors = {a.swap_format}; rd.vertex_stride = 12; rd.attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
  Pipeline road_p = make_graphics(a, rd, "road");
  GfxDesc hd; hd.vs = hd.fs = sh; hd.vs_entry = "vsHud"; hd.fs_entry = "fsHud";
  hd.colors = {a.swap_format}; hd.vertex_stride = 8; hd.attributes = {{0, 0, VK_FORMAT_R32G32_SFLOAT, 0}};
  Pipeline hud_p = make_graphics(a, hd, "hud_plan");

  run_frames(a, frames_from_env(3), 2, [&](VkCommandBuffer c, uint32_t idx, int f) {
    begin_rendering(a, c, {a.swap_views[idx]});
    VkDeviceSize off = 0;
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, road_p.pipe);
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, road_p.layout, 0, 1, &set, 0, nullptr);
    vkCmdBindVertexBuffers(c, 0, 1, &road.buf, &off);
    vkCmdDraw(c, (uint32_t)(ribbon.size() / 3), 1, 0, 0);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, hud_p.pipe);
    vkCmdBindVertexBuffers(c, 0, 1, &hud.buf, &off);
    vkCmdDraw(c, (uint32_t)(plan.size() / 2), 1, 0, 0);
    vkCmdEndRendering(c);
  });
}
