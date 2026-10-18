import covenant;
// Two chained dispatches of a GPU prefix scan (multi-dispatch kernel):
// pass 1 scans each 64-element block and writes the block totals,
// pass 2 adds the scanned block totals back into every element.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> data;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> blockSums;

groupshared uint tile[64];

[shader("compute")] [numthreads(64, 1, 1)]
void scanBlocks(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint i = gid.x * 64 + lid.x;
    tile[lid.x] = data[i];
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < 64; off <<= 1) {
        uint v = lid.x >= off ? tile[lid.x - off] : 0;
        GroupMemoryBarrierWithGroupSync();
        tile[lid.x] += v;
        GroupMemoryBarrierWithGroupSync();
    }
    data[i] = tile[lid.x];
    if (lid.x == 63) blockSums[gid.x] = tile[63];
}

[shader("compute")] [numthreads(64, 1, 1)]
void addBlockSums(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint prefix = 0;
    for (uint b = 0; b < gid.x; b++) prefix += blockSums[b];
    data[gid.x * 64 + lid.x] += prefix;
}
