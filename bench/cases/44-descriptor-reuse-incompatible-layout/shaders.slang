import covenant;
[[vk::binding(0, 0)]] StructuredBuffer<float> inputValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outputValues;

[shader("compute")] [numthreads(64, 1, 1)]
void sumMain(uint3 id : SV_DispatchThreadID) {
    outputValues[id.x] = inputValues[id.x] + outputValues[id.x];
}

struct ScaleParams { float scale; float bias; float2 pad; };
[[vk::push_constant]] ConstantBuffer<ScaleParams> params;
[shader("compute")] [numthreads(64, 1, 1)]
void scaleMain(uint3 id : SV_DispatchThreadID) {
    outputValues[id.x] = inputValues[id.x] * params.scale + params.bias;
}
