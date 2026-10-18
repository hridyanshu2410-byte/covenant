import covenant;
struct VOut { float4 pos : SV_Position; };
struct FillPush { float4 color; float4 pad; };
[[vk::push_constant]] FillPush fillPush;
// five-point star path, emitted as fan triangles (0, k, k+1)
[shader("vertex")]
VOut vsPath(uint vid : SV_VertexID) {
    uint tri = vid / 3, corner = vid % 3;
    uint k = corner == 0 ? 0 : (tri + corner) % 5;
    float ang = 6.2831853 * float(k * 2 % 5) / 5.0 - 1.5707963;
    VOut o; o.pos = float4(0.8 * cos(ang), 0.8 * sin(ang), 0, 1);
    return o;
}
[shader("fragment")]
float4 fsStencil(VOut i) : SV_Target { return float4(0, 0, 0, 0); }
[shader("fragment")]
float4 fsCover(VOut i) : SV_Target { return fillPush.color; }
