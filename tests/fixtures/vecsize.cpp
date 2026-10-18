// A storage buffer sized with the host vector's sizeof (12 bytes for a uvec3),
// for a shader array whose std430 stride is 16 (Khronos oit_linked_lists, KS-6).
#include <glm/glm.hpp>
#include <cstdint>
uint64_t fragment_buffer_size(uint32_t count) {
#ifdef FIXED
    return sizeof(glm::uvec4) * count;
#else
    return sizeof(glm::uvec3) * count;
#endif
}
