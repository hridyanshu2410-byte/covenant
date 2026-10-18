// Distilled from randomgraphics/rapid-vulkan PR #56: the command recorder
// caches descriptor sets and skips allocating and binding a new one when a
// draw or dispatch's resource writes match the set already bound. Nothing
// checked that the pipeline layout was the same, so two consecutive
// dispatches with identical storage-buffer bindings but different push
// constant layouts shared the set bound with the first layout; pipeline
// layouts with different push constant ranges are not compatible for any set
// (VUID-vkCmdDispatch-None-08600). The fix adds a guard: reuse only when
// the pipeline layout handle is the same (and makes the pool-key comparison
// inspect every binding, which this reproduction does not exercise).
#include "../../common/app.hpp"
using namespace bench;

struct ComputeProgram { VkPipeline pipeline; VkPipelineLayout layout; VkDescriptorSetLayout set_layout; uint32_t push_size; };

static ComputeProgram make_program(App& a, VkShaderModule sh, const char* entry, uint32_t push_size, const std::string& nm) {
  ComputeProgram p{}; p.push_size = push_size;
  p.set_layout = make_set_layout(a, {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
                                     {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}});
  VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1; li.pSetLayouts = &p.set_layout;
  li.pushConstantRangeCount = push_size ? 1 : 0; li.pPushConstantRanges = &pr;
  VK(vkCreatePipelineLayout(a.dev, &li, nullptr, &p.layout));
  VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = sh; ci.stage.pName = entry;
  ci.layout = p.layout;
  VK(vkCreateComputePipelines(a.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline));
  a.name(p.pipeline, VK_OBJECT_TYPE_PIPELINE, nm);
  return p;
}

// CommandRecorder's descriptor cache
struct Recorder {
  App* a; VkCommandBuffer cmd;
  bool have_bound = false;
  VkPipelineLayout bound_layout = VK_NULL_HANDLE;
  VkBuffer bound_writes[2] = {};

  void dispatch(const ComputeProgram& prog, VkBuffer in, VkBuffer out, const void* push, uint32_t groups) {
    bool same_writes = have_bound && bound_writes[0] == in && bound_writes[1] == out;
#ifdef FIXED
    bool reuse = same_writes && bound_layout == prog.layout;   // conservative layout-handle equality
#else
    bool reuse = same_writes;
#endif
    if (!reuse) {
      VkDescriptorSet s = alloc_set(*a, prog.set_layout, "compute_args");
      Buffer bi{in, VK_NULL_HANDLE, nullptr, VK_WHOLE_SIZE}, bo{out, VK_NULL_HANDLE, nullptr, VK_WHOLE_SIZE};
      write_buffer(*a, s, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bi);
      write_buffer(*a, s, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bo);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prog.layout, 0, 1, &s, 0, nullptr);
      have_bound = true; bound_layout = prog.layout; bound_writes[0] = in; bound_writes[1] = out;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prog.pipeline);
    if (prog.push_size) vkCmdPushConstants(cmd, prog.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, prog.push_size, push);
    vkCmdDispatch(cmd, groups, 1, 1);
  }
};

int main(int argc, char** argv) {
  App a = make_app(argc, argv);
  const uint32_t N = 4096;
  Buffer input = make_buffer(a, N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "input_values");
  Buffer output = make_buffer(a, N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "output_values");
  float* in = (float*)input.map; for (uint32_t i = 0; i < N; i++) in[i] = (float)i;
  auto sh = load_shader(a, "shaders.spv");
  ComputeProgram sum = make_program(a, sh, "sumMain", 0, "sum_values");
  ComputeProgram scale = make_program(a, sh, "scaleMain", 16, "scale_values");

  VkCommandBuffer c = alloc_cmd(a, a.comp_pool);
  begin(c);
  Recorder rec{&a, c};
  rec.dispatch(sum, input.buf, output.buf, nullptr, N / 64);
  buffer_barrier(c, output.buf, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  float scale_params[4] = {0.5f, 1.0f, 0, 0};
  rec.dispatch(scale, input.buf, output.buf, scale_params, N / 64);
  buffer_barrier(c, output.buf, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
  VK(vkEndCommandBuffer(c));
  VkFence f = make_fence(a, false, "compute_fence");
  submit(a.comp, c, {}, {}, f);
  VK(vkWaitForFences(a.dev, 1, &f, VK_TRUE, UINT64_MAX));
  vkDeviceWaitIdle(a.dev);
  return 0;
}
