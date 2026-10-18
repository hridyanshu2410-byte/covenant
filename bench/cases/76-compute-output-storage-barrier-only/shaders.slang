import covenant;
// GPU-driven rendering (bgfx example 37-gpudrivenrendering): a compute pass
// culls the object list, writes the surviving instances' data and fills
// the indexed-indirect draw arguments; the draw then consumes both.
struct ObjectInfo { float4 boundsCenterRadius; float4 offsetScale; };
struct DrawIndexedIndirect { uint indexCount; uint instanceCount; uint firstIndex; int vertexOffset; uint firstInstance; };
struct InstanceData { float4 offsetScale; };

[[vk::binding(0, 0)]] StructuredBuffer<ObjectInfo> objects;
[[vk::binding(1, 0)]] RWStructuredBuffer<InstanceData> instanceOut;
[[vk::binding(2, 0)]] RWStructuredBuffer<DrawIndexedIndirect> drawArgs;

[shader("compute")] [numthreads(64, 1, 1)]
void csCull(uint3 id : SV_DispatchThreadID) {
    if (id.x == 0) { drawArgs[0].indexCount = 6; drawArgs[0].firstIndex = 0; drawArgs[0].vertexOffset = 0; drawArgs[0].firstInstance = 0; }
    ObjectInfo o = objects[id.x];
    bool visible = abs(o.boundsCenterRadius.x) - o.boundsCenterRadius.w < 1.0;   // inside the frustum's x range
    if (visible) {
        uint slot;
        InterlockedAdd(drawArgs[0].instanceCount, 1, slot);
        instanceOut[slot].offsetScale = o.offsetScale;
    }
}

struct VIn { float2 pos : POSITION; float4 inst : INSTANCE; };
struct VOut { float4 pos : SV_Position; };
[shader("vertex")]
VOut vsMain(VIn v) { VOut o; o.pos = float4(v.pos * v.inst.zw + v.inst.xy, 0.5, 1); return o; }
[shader("fragment")]
float4 fsMain() : SV_Target { return float4(0.2, 0.7, 0.3, 1); }
