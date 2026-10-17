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

template [[host_name("copy_f32")]] kernel void
copy_strided<float>(device const float *, device float *,
                    constant StridedArgs &, uint);
template [[host_name("copy_bf16")]] kernel void
copy_strided<bf16_t>(device const bf16_t *, device bf16_t *,
                     constant StridedArgs &, uint);
template [[host_name("copy_u32")]] kernel void
copy_strided<uint>(device const uint *, device uint *, constant StridedArgs &,
                   uint);

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
