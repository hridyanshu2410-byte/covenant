import covenant;
// PlayCanvas gsplat vertex shader (reduced): projects each splat's 3D
// covariance to screen space (EWA, J = Jacobian of the perspective
// projection in pixels) and expands a quad along the 2D covariance's axes.
struct Splat { [Frame("world")] float4 center; float4 covA; float4 covB; float4 color; };   // cov: xx xy xz . / yy yz zz .
struct View {
    [Maps("world","view")] float4x4 view;
    [Maps("view","clip")] float4x4 proj;
    float2 viewport_size; float2 pad;
};
[[vk::binding(0, 0)]] StructuredBuffer<Splat> splats;
[[vk::binding(1, 0)]] ConstantBuffer<View> v;

struct VOut { float4 pos : SV_Position; float2 offset : OFFSET; float4 color : COLOR; };

[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
    Splat s = splats[iid];
    float4 vc = mul(v.view, float4(s.center.xyz, 1));
    float4 cc = mul(v.proj, vc);
    float projMat00 = v.proj[0][0];
    float projMat11 = v.proj[1][1];
    float focal = v.viewport_size.x * projMat00;
#ifdef FIXED
    float focalX = abs(v.viewport_size.x * projMat00);
    float focalY = abs(v.viewport_size.y * projMat11);
#else
    float focalX = focal, focalY = focal;
#endif
    float J1x = focalX / vc.z, J1y = focalY / vc.z;
    float2 J2 = -float2(J1x, J1y) / vc.z * vc.xy;
    float3x3 J = float3x3(J1x, 0.0, J2.x,
                          0.0, J1y, J2.y,
                          0.0, 0.0, 0.0);
    float3x3 W = (float3x3)v.view;
    float3x3 Vrk = float3x3(s.covA.x, s.covA.y, s.covA.z,
                            s.covA.y, s.covB.x, s.covB.y,
                            s.covA.z, s.covB.y, s.covB.z);
    float3x3 T = mul(J, W);
    float3x3 cov = mul(mul(T, Vrk), transpose(T));
    float a = cov[0][0] + 0.3, d = cov[1][1] + 0.3, b = cov[0][1];
    float mid = 0.5 * (a + d), radius = length(float2((a - d) * 0.5, b));
    float l1 = mid + radius, l2 = max(mid - radius, 0.1);
    float2 diag = normalize(float2(b, l1 - a) + float2(1e-6, 0));
    float2 v1 = sqrt(2.0 * l1) * diag, v2 = sqrt(2.0 * l2) * float2(diag.y, -diag.x);
    float2 corner = float2((vid << 1) & 2, vid & 2) - 1.0;         // one triangle covering the quad
    float2 offsetPx = corner.x * v1 + corner.y * v2;
    VOut o;
    o.pos = cc + float4(offsetPx / v.viewport_size * 2.0 * cc.w, 0, 0);
    o.offset = corner * 2.0;
    o.color = s.color;
    return o;
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target {
    float A = dot(i.offset, i.offset);
    if (A > 4.0) discard;
    float B = exp(-A * 4.0) * i.color.a;
    return float4(i.color.rgb * B, B);
}
