import covenant;
// A packed sRGB colour unpacked by hand (shift, mask, convert: the manual
// unpackUnorm4x8) reaches an sRGB attachment, which expects linear values;
// the fix linearizes. The bit fields keep the packed word's encoding.
struct Uniforms { [Color("srgb")] uint tint; };
[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> u;
float4 unpack(uint c) {
    return float4(float(c & 0xff), float((c >> 8) & 0xff), float((c >> 16) & 0xff), float(c >> 24)) / 255.0;
}
float3 linearize(float3 s) {
    return select(s <= 0.04045, s / 12.92, pow((s + 0.055) / 1.055, 2.4));
}
[shader("fragment")]
float4 fsBug() : SV_Target { return unpack(u.tint); }
[shader("fragment")]
float4 fsFixed() : SV_Target { float4 c = unpack(u.tint); return float4(linearize(c.rgb), c.a); }
