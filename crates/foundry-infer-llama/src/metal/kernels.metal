#include <metal_stdlib>
using namespace metal;

constant constexpr uint GROUP = 64;
constant constexpr uint HEAD_DIM = 128;
constant constexpr uint HALF_DIM = 64;
constant constexpr float SENTINEL = -1.0e30f;

struct EmbedParams {
    uint rows;
    uint hidden;
};

kernel void embed_gather(
    device const uint* ids [[buffer(0)]],
    device const uint* weight [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* biases [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant EmbedParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint words = p.hidden / 8;
    uint word = gid.x;
    uint row = gid.y;
    if (word >= words || row >= p.rows) {
        return;
    }
    ulong token = ids[row];
    uint packed = weight[token * words + word];
    ulong group = token * (p.hidden / GROUP) + word / 8;
    float scale = scales[group];
    float bias = biases[group];
    device half* destination = out + ulong(row) * p.hidden + word * 8;
    for (uint k = 0; k < 8; ++k) {
        destination[k] = half(scale * float((packed >> (4 * k)) & 0xF) + bias);
    }
}

struct NormParams {
    uint dim;
    float eps;
};

kernel void rms_norm(
    device const half* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant NormParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[32];
    device const half* source = x + ulong(row) * p.dim;
    device half* destination = out + ulong(row) * p.dim;
    float sum = 0.0f;
    for (uint i = tid; i < p.dim; i += threads) {
        float value = source[i];
        sum += value * value;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partial[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    uint simdgroups = (threads + 31) / 32;
    for (uint i = 0; i < simdgroups; ++i) {
        total += partial[i];
    }
    float inverse = metal::precise::rsqrt(total / float(p.dim) + p.eps);
    for (uint i = tid; i < p.dim; i += threads) {
        destination[i] = weight[i] * half(float(source[i]) * inverse);
    }
}

struct MatmulParams {
    uint rows;
    uint in_features;
    uint out_features;
};

constant constexpr uint QMV_ROWS = 4;
constant constexpr uint QMV_SIMDGROUPS = 2;

template <bool RESIDUAL>
kernel void qmv(
    device const half* x [[buffer(0)]],
    device const uint* weight [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* biases [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant MatmulParams& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    uint first = (tg.x * QMV_SIMDGROUPS + simdgroup) * QMV_ROWS;
    if (first >= p.out_features) {
        return;
    }
    uint words = p.in_features / 8;
    uint groups = p.in_features / GROUP;
    device const half* input = x + ulong(tg.y) * p.in_features;
    float acc[QMV_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint base = lane * 4; base < words; base += 128) {
        float xs[32];
        float xsum = 0.0f;
        device const half4* xv = (device const half4*)(input + base * 8);
        for (uint i = 0; i < 8; ++i) {
            float4 v = float4(xv[i]);
            xs[4 * i] = v.x;
            xs[4 * i + 1] = v.y;
            xs[4 * i + 2] = v.z;
            xs[4 * i + 3] = v.w;
            xsum += v.x + v.y + v.z + v.w;
        }
        uint group = base / 8;
        for (uint r = 0; r < QMV_ROWS; ++r) {
            ulong row = first + r;
            uint4 packed = *((device const uint4*)(weight + row * words + base));
            float dot = 0.0f;
            for (uint j = 0; j < 4; ++j) {
                uint word = packed[j];
                for (uint k = 0; k < 8; ++k) {
                    dot += xs[8 * j + k] * float((word >> (4 * k)) & 0xF);
                }
            }
            float scale = scales[row * groups + group];
            float bias = biases[row * groups + group];
            acc[r] += scale * dot + bias * xsum;
        }
    }
    for (uint r = 0; r < QMV_ROWS; ++r) {
        float total = simd_sum(acc[r]);
        if (lane == 0) {
            ulong index = ulong(tg.y) * p.out_features + first + r;
            half value = half(total);
            out[index] = RESIDUAL ? half(float(out[index]) + float(value)) : value;
        }
    }
}

template [[host_name("qmv")]] kernel void qmv<false>(
    device const half*, device const uint*, device const half*, device const half*,
    device half*, constant MatmulParams&, uint2, uint, uint);
template [[host_name("qmv_residual")]] kernel void qmv<true>(
    device const half*, device const uint*, device const half*, device const half*,
    device half*, constant MatmulParams&, uint2, uint, uint);

constant constexpr uint QMM_BM = 64;
constant constexpr uint QMM_BN = 64;
constant constexpr uint QMM_BK = 64;
constant constexpr uint QMM_LD = QMM_BK + 8;
constant constexpr uint QMM_THREADS = 128;

template <bool RESIDUAL>
kernel void qmm(
    device const half* x [[buffer(0)]],
    device const uint* weight [[buffer(1)]],
    device const half* scales [[buffer(2)]],
    device const half* biases [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant MatmulParams& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup half shared[(QMM_BM + QMM_BN) * QMM_LD];
    threadgroup half* xs = shared;
    threadgroup half* ws = shared + QMM_BM * QMM_LD;
    uint m0 = tg.y * QMM_BM;
    uint n0 = tg.x * QMM_BN;
    uint sm = (simdgroup / 2) * 32;
    uint sn = (simdgroup % 2) * 32;
    uint words = p.in_features / 8;
    uint groups = p.in_features / GROUP;
    simdgroup_matrix<float, 8, 8> acc[4][4];
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    for (uint k0 = 0; k0 < p.in_features; k0 += QMM_BK) {
        for (uint i = tid; i < QMM_BM * QMM_BK / 4; i += QMM_THREADS) {
            uint r = i / (QMM_BK / 4);
            uint c = (i % (QMM_BK / 4)) * 4;
            uint row = m0 + r;
            half4 value = row < p.rows
                ? *((device const half4*)(x + ulong(row) * p.in_features + k0 + c))
                : half4(0.0h);
            *((threadgroup half4*)(xs + r * QMM_LD + c)) = value;
        }
        for (uint i = tid; i < QMM_BN * (QMM_BK / 8); i += QMM_THREADS) {
            uint r = i / (QMM_BK / 8);
            uint w = i % (QMM_BK / 8);
            ulong column = n0 + r;
            uint packed = weight[column * words + k0 / 8 + w];
            float scale = scales[column * groups + k0 / GROUP];
            float bias = biases[column * groups + k0 / GROUP];
            threadgroup half* destination = ws + r * QMM_LD + w * 8;
            for (uint k = 0; k < 8; ++k) {
                destination[k] = half(scale * float((packed >> (4 * k)) & 0xF) + bias);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < QMM_BK; kk += 8) {
            simdgroup_matrix<half, 8, 8> a[4];
            simdgroup_matrix<half, 8, 8> b[4];
            for (uint i = 0; i < 4; ++i) {
                simdgroup_load(a[i], xs + (sm + 8 * i) * QMM_LD + kk, QMM_LD);
                simdgroup_load(b[i], ws + (sn + 8 * i) * QMM_LD + kk, QMM_LD, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; ++i) {
                for (uint j = 0; j < 4; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* cs = (threadgroup float*)shared;
    for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) {
            simdgroup_store(acc[i][j], cs + (sm + 8 * i) * QMM_BN + sn + 8 * j, QMM_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < QMM_BM * QMM_BN; i += QMM_THREADS) {
        uint r = i / QMM_BN;
        uint c = i % QMM_BN;
        uint row = m0 + r;
        if (row < p.rows) {
            ulong index = ulong(row) * p.out_features + n0 + c;
            half value = half(cs[i]);
            out[index] = RESIDUAL ? half(float(out[index]) + float(value)) : value;
        }
    }
}

template [[host_name("qmm")]] kernel void qmm<false>(
    device const half*, device const uint*, device const half*, device const half*,
    device half*, constant MatmulParams&, uint2, uint, uint);
template [[host_name("qmm_residual")]] kernel void qmm<true>(
    device const half*, device const uint*, device const half*, device const half*,
    device half*, constant MatmulParams&, uint2, uint, uint);

struct RopeParams {
    uint rows;
    uint offset;
    uint heads;
    uint kv_heads;
    uint capacity;
};

kernel void rope_store(
    device half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half* k_cache [[buffer(3)]],
    device half* v_cache [[buffer(4)]],
    device const float* frequencies [[buffer(5)]],
    constant RopeParams& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint i = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    if (i >= HALF_DIM || head >= p.heads + 2 * p.kv_heads || row >= p.rows) {
        return;
    }
    uint position = p.offset + row;
    device const half* source;
    device half* destination;
    bool rotate = true;
    if (head < p.heads) {
        source = q + (ulong(row) * p.heads + head) * HEAD_DIM;
        destination = q + (ulong(row) * p.heads + head) * HEAD_DIM;
    } else if (head < p.heads + p.kv_heads) {
        uint kv = head - p.heads;
        source = k + (ulong(row) * p.kv_heads + kv) * HEAD_DIM;
        destination = k_cache + (ulong(kv) * p.capacity + position) * HEAD_DIM;
    } else {
        uint kv = head - p.heads - p.kv_heads;
        source = v + (ulong(row) * p.kv_heads + kv) * HEAD_DIM;
        destination = v_cache + (ulong(kv) * p.capacity + position) * HEAD_DIM;
        rotate = false;
    }
    float x1 = source[i];
    float x2 = source[i + HALF_DIM];
    if (rotate) {
        float inverse = metal::precise::divide(1.0f, frequencies[i]);
        float theta = float(position) * inverse;
        float c = metal::precise::cos(theta);
        float s = metal::precise::sin(theta);
        destination[i] = half(x1 * c - x2 * s);
        destination[i + HALF_DIM] = half(x1 * s + x2 * c);
    } else {
        destination[i] = half(x1);
        destination[i + HALF_DIM] = half(x2);
    }
}

struct AttentionParams {
    uint rows;
    uint offset;
    uint heads;
    uint kv_heads;
    uint capacity;
    float scale;
};

constant constexpr uint ATTENTION_SIMDGROUPS = 8;

kernel void attention(
    device const half* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant AttentionParams& p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup float maxima[ATTENTION_SIMDGROUPS];
    threadgroup float sums[ATTENTION_SIMDGROUPS];
    threadgroup float4 partial[ATTENTION_SIMDGROUPS][32];
    uint head = tg.x;
    uint row = tg.y;
    uint kv = head / (p.heads / p.kv_heads);
    uint last = p.offset + row;
    ulong q_index = (ulong(row) * p.heads + head) * HEAD_DIM + lane * 4;
    float4 query = float4(*((device const half4*)(q + q_index))) * p.scale;
    device const half* keys = k_cache + ulong(kv) * p.capacity * HEAD_DIM + lane * 4;
    device const half* values = v_cache + ulong(kv) * p.capacity * HEAD_DIM + lane * 4;
    float maximum = SENTINEL;
    float sum = 0.0f;
    float4 acc = float4(0.0f);
    for (uint j = simdgroup; j <= last; j += ATTENTION_SIMDGROUPS) {
        float4 key = float4(*((device const half4*)(keys + ulong(j) * HEAD_DIM)));
        float score = simd_sum(dot(query, key));
        float updated = max(maximum, score);
        float correction = exp(maximum - updated);
        float weight = exp(score - updated);
        float4 value = float4(*((device const half4*)(values + ulong(j) * HEAD_DIM)));
        sum = sum * correction + weight;
        acc = acc * correction + weight * value;
        maximum = updated;
    }
    if (lane == 0) {
        maxima[simdgroup] = maximum;
        sums[simdgroup] = sum;
    }
    partial[simdgroup][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup != 0) {
        return;
    }
    float global = SENTINEL;
    for (uint i = 0; i < ATTENTION_SIMDGROUPS; ++i) {
        if (sums[i] > 0.0f) {
            global = max(global, maxima[i]);
        }
    }
    float total = 0.0f;
    float4 combined = float4(0.0f);
    for (uint i = 0; i < ATTENTION_SIMDGROUPS; ++i) {
        if (sums[i] > 0.0f) {
            float correction = exp(maxima[i] - global);
            total += sums[i] * correction;
            combined += partial[i][lane] * correction;
        }
    }
    *((device half4*)(out + q_index)) = half4(combined / total);
}

struct ElementwiseParams {
    uint count;
};

kernel void swiglu(
    device half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= p.count) {
        return;
    }
    float g = gate[index];
    half sigmoid = half(1.0f / (1.0f + metal::precise::exp(-g)));
    half silu = half(g * float(sigmoid));
    gate[index] = half(float(silu) * float(up[index]));
}

struct ArgmaxParams {
    uint count;
    uint slot;
};

constant constexpr uint ARGMAX_THREADS = 1024;

kernel void argmax(
    device const half* logits [[buffer(0)]],
    device uint* outputs [[buffer(1)]],
    device uint* next [[buffer(2)]],
    constant ArgmaxParams& p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reduce_value[32];
    threadgroup uint reduce_index[32];
    float maximum = SENTINEL;
    for (uint i = tid; i < p.count; i += ARGMAX_THREADS) {
        maximum = max(maximum, float(logits[i]));
    }
    maximum = simd_max(maximum);
    if (lane == 0) {
        reduce_value[simdgroup] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = SENTINEL;
    for (uint i = 0; i < ARGMAX_THREADS / 32; ++i) {
        maximum = max(maximum, reduce_value[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = tid; i < p.count; i += ARGMAX_THREADS) {
        sum += metal::precise::exp(float(logits[i]) - maximum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        reduce_value[simdgroup] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.0f;
    for (uint i = 0; i < ARGMAX_THREADS / 32; ++i) {
        sum += reduce_value[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float normalizer = float(half(maximum + metal::precise::log(sum)));
    float best = SENTINEL;
    uint best_index = 0xFFFFFFFF;
    for (uint i = tid; i < p.count; i += ARGMAX_THREADS) {
        float value = float(half(float(logits[i]) - normalizer));
        if (value > best) {
            best = value;
            best_index = i;
        }
    }
    for (uint offset = 16; offset > 0; offset /= 2) {
        float other = simd_shuffle_down(best, offset);
        uint other_index = simd_shuffle_down(best_index, offset);
        if (other > best || (other == best && other_index < best_index)) {
            best = other;
            best_index = other_index;
        }
    }
    if (lane == 0) {
        reduce_value[simdgroup] = best;
        reduce_index[simdgroup] = best_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float winner = SENTINEL;
        uint winner_index = 0xFFFFFFFF;
        for (uint i = 0; i < ARGMAX_THREADS / 32; ++i) {
            float value = reduce_value[i];
            uint index = reduce_index[i];
            if (value > winner || (value == winner && index < winner_index)) {
                winner = value;
                winner_index = index;
            }
        }
        outputs[p.slot] = winner_index;
        next[0] = winner_index;
    }
}
