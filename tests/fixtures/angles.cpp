// Host angle typing: degrees from a 180/pi factor reach std::cos through a
// helper (bug); a polymorphic helper used with both units is not a conflict.
#include <cmath>
static const float kPi = 3.14159265f, DEG_TO_RAD = kPi / 180.0f, RAD_TO_DEG = 180.0f / kPi;
struct Spot { float cone; };                       // degrees
static float falloff(float angle) { return std::cos(angle); }   // radians
static float twice(float x) { return x * 2.0f; }
float run(float coneRad) {
  Spot s{coneRad * RAD_TO_DEG};
  float a = twice(s.cone), b = twice(coneRad);     // polymorphic helper: fine
  (void)a; (void)b;
#ifdef FIXED
  return falloff(s.cone * DEG_TO_RAD);
#else
  return falloff(s.cone);
#endif
}
