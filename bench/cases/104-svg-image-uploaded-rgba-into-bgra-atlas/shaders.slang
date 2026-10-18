import covenant;
// gpui polychrome sprite pipeline, reduced: each sprite is a rectangle in
// window pixels sampling one tile of the BGRA8 polychrome atlas.
struct GlobalParams { float2 viewport_size; float2 pad; };
[[vk::binding(0, 0)]] ConstantBuffer<GlobalParams> globals;

struct PolychromeSprite {
    float4 bounds;        // origin.xy, size.xy (pixels)
    float4 tile_uv;       // atlas uv origin.xy, size.xy
    float opacity;
    uint grayscale;
    float2 pad;
};
[[vk::binding(1, 0)]] StructuredBuffer<PolychromeSprite> b_poly_sprites;
[[vk::binding(2, 0)]] [Color("srgb")] Sampler2D t_sprite;   // polychrome atlas, straight alpha

struct VOut {
    float4 position : SV_Position;
    float2 tile_position : TEXCOORD0;
    nointerpolation uint sprite_id : TEXCOORD1;
};

[shader("vertex")]
VOut vs_poly_sprite(uint vertex_id : SV_VertexID, uint sprite_id : SV_InstanceID) {
    static const float2 corners[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(0, 1), float2(1, 0), float2(1, 1) };
    float2 unit_vertex = corners[vertex_id];
    PolychromeSprite sprite = b_poly_sprites[sprite_id];
    float2 position = unit_vertex * sprite.bounds.zw + sprite.bounds.xy;
    VOut o;
    o.position = float4(position / globals.viewport_size * float2(2.0, 2.0) - float2(1.0, 1.0), 0.0, 1.0);
    o.tile_position = sprite.tile_uv.xy + unit_vertex * sprite.tile_uv.zw;
    o.sprite_id = sprite_id;
    return o;
}

[shader("fragment")]
float4 fs_poly_sprite(VOut input) : SV_Target {
    PolychromeSprite sprite = b_poly_sprites[input.sprite_id];
    float4 sample = t_sprite.Sample(input.tile_position);
    float4 color = sample;
    if (sprite.grayscale != 0) {
        float grayscale = dot(color.rgb, float3(0.2126, 0.7152, 0.0722));
        color = float4(grayscale, grayscale, grayscale, color.a);
    }
    // premultiplied output
    return float4(color.rgb * color.a * sprite.opacity, color.a * sprite.opacity);
}
