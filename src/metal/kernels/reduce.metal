struct ReduceArgs {
  StridedArgs rows;
  uint dim_size;
  uint dim_stride;
};

struct SumOp {
  static float identity() { return 0.0f; }
  static float apply(float lhs, float rhs) { return lhs + rhs; }
};

struct AmaxOp {
  static float identity() { return -INFINITY; }
  static float apply(float lhs, float rhs) { return max(lhs, rhs); }
};

template <typename T, typename Op>
kernel void reduce_rows(device const T *input [[buffer(0)]],
                        device T *output [[buffer(1)]],
                        constant ReduceArgs &args [[buffer(2)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float partial[REDUCE_THREADS];
  uint base = strided_index(row, args.rows);
  float acc = Op::identity();
  for (uint k = tid; k < args.dim_size; k += REDUCE_THREADS) {
    acc = Op::apply(acc, to_float(input[base + k * args.dim_stride]));
  }
  partial[tid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint width = REDUCE_THREADS / 2; width > 0; width /= 2) {
    if (tid < width) {
      partial[tid] = Op::apply(partial[tid], partial[tid + width]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    output[row] = from_float<T>(partial[0]);
  }
}

template <typename T>
kernel void argmax_rows(device const T *input [[buffer(0)]],
                        device uint *output [[buffer(1)]],
                        constant ReduceArgs &args [[buffer(2)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float values[REDUCE_THREADS];
  threadgroup uint indices[REDUCE_THREADS];
  uint base = strided_index(row, args.rows);
  float best = -INFINITY;
  uint best_index = 0;
  for (uint k = tid; k < args.dim_size; k += REDUCE_THREADS) {
    float value = to_float(input[base + k * args.dim_stride]);
    if (value > best) {
      best = value;
      best_index = k;
    }
  }
  values[tid] = best;
  indices[tid] = best_index;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint width = REDUCE_THREADS / 2; width > 0; width /= 2) {
    if (tid < width) {
      float other = values[tid + width];
      uint other_index = indices[tid + width];
      if (other > values[tid] ||
          (other == values[tid] && other_index < indices[tid])) {
        values[tid] = other;
        indices[tid] = other_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    output[row] = indices[0];
  }
}

#define INSTANTIATE_REDUCE(name, T, Op)                                        \
  template [[host_name(name)]] kernel void reduce_rows<T, Op>(                 \
      device const T *, device T *, constant ReduceArgs &, uint, uint);

#define INSTANTIATE_ARGMAX(name, T)                                            \
  template [[host_name(name)]] kernel void argmax_rows<T>(                     \
      device const T *, device uint *, constant ReduceArgs &, uint, uint);

INSTANTIATE_REDUCE("reduce_sum_f32", float, SumOp)
INSTANTIATE_REDUCE("reduce_sum_bf16", bf16_t, SumOp)
INSTANTIATE_REDUCE("reduce_amax_f32", float, AmaxOp)
INSTANTIATE_REDUCE("reduce_amax_bf16", bf16_t, AmaxOp)
INSTANTIATE_ARGMAX("reduce_argmax_f32", float)
INSTANTIATE_ARGMAX("reduce_argmax_bf16", bf16_t)
