#include <metal_stdlib>

using namespace metal;

constant uint CHUNK = 64;

kernel void checksum(
    device const uchar *weights [[buffer(0)]],
    device atomic_uint *result [[buffer(1)]],
    constant uint &length [[buffer(2)]],
    uint chunk [[thread_position_in_grid]])
{
    uint start = chunk * CHUNK;
    if (start >= length) {
        return;
    }
    uint end = start + min(CHUNK, length - start);
    uint sum = 0;
    for (uint index = start; index < end; ++index) {
        sum += uint(weights[index]) * (index + 1);
    }
    atomic_fetch_add_explicit(result, sum * 2654435761u + chunk, memory_order_relaxed);
}
