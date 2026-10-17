template <typename T>
kernel void copy_strided(device const T *input [[buffer(0)]],
                         device T *output [[buffer(1)]],
                         constant StridedArgs &args [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= args.count) {
    return;
  }
  output[gid] = input[strided_index(gid, args)];
}

#define INSTANTIATE_COPY(name, T)                                              \
  template [[host_name(name)]] kernel void copy_strided<T>(                    \
      device const T *, device T *, constant StridedArgs &, uint);

INSTANTIATE_COPY("copy_f32", float)
INSTANTIATE_COPY("copy_bf16", bf16_t)
INSTANTIATE_COPY("copy_u32", uint)

template <typename In, typename Out>
kernel void to_dtype_strided(device const In *input [[buffer(0)]],
                             device Out *output [[buffer(1)]],
                             constant StridedArgs &args [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
  if (gid >= args.count) {
    return;
  }
  output[gid] = from_float<Out>(to_float(input[strided_index(gid, args)]));
}

#define INSTANTIATE_TO_DTYPE(name, In, Out)                                    \
  template [[host_name(name)]] kernel void to_dtype_strided<In, Out>(          \
      device const In *, device Out *, constant StridedArgs &, uint);

INSTANTIATE_TO_DTYPE("to_dtype_f32_bf16", float, bf16_t)
INSTANTIATE_TO_DTYPE("to_dtype_bf16_f32", bf16_t, float)
