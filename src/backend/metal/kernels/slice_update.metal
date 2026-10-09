template <typename T>
kernel void slice_update_strided(device const T *input [[buffer(0)]],
                                 device T *output [[buffer(1)]],
                                 constant StridedArgs &input_args [[buffer(2)]],
                                 constant StridedArgs &output_args
                                 [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= input_args.count) {
    return;
  }
  output[strided_index(gid, output_args)] =
      input[strided_index(gid, input_args)];
}

#define INSTANTIATE_SLICE_UPDATE(name, T)                                      \
  template [[host_name(name)]] kernel void slice_update_strided<T>(            \
      device const T *, device T *, constant StridedArgs &,                    \
      constant StridedArgs &, uint);

INSTANTIATE_SLICE_UPDATE("slice_update_f32", float)
INSTANTIATE_SLICE_UPDATE("slice_update_f16", half)
INSTANTIATE_SLICE_UPDATE("slice_update_bf16", bf16_t)
