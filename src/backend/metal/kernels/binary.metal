struct AddOp {
  static float apply(float lhs, float rhs) { return lhs + rhs; }
};

struct SubOp {
  static float apply(float lhs, float rhs) { return lhs - rhs; }
};

struct MulOp {
  static float apply(float lhs, float rhs) { return lhs * rhs; }
};

struct DivOp {
  static float apply(float lhs, float rhs) { return lhs / rhs; }
};

template <typename T, typename Op>
kernel void binary_strided(device const T *lhs [[buffer(0)]],
                           device const T *rhs [[buffer(1)]],
                           device T *output [[buffer(2)]],
                           constant StridedArgs &lhs_args [[buffer(3)]],
                           constant StridedArgs &rhs_args [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= lhs_args.count) {
    return;
  }
  output[gid] =
      from_float<T>(Op::apply(to_float(lhs[strided_index(gid, lhs_args)]),
                              to_float(rhs[strided_index(gid, rhs_args)])));
}

#define INSTANTIATE_BINARY(name, T, Op)                                        \
  template [[host_name(name)]] kernel void binary_strided<T, Op>(              \
      device const T *, device const T *, device T *, constant StridedArgs &,  \
      constant StridedArgs &, uint);

INSTANTIATE_BINARY("binary_add_f32", float, AddOp)
INSTANTIATE_BINARY("binary_add_f16", half, AddOp)
INSTANTIATE_BINARY("binary_add_bf16", bf16_t, AddOp)
INSTANTIATE_BINARY("binary_sub_f32", float, SubOp)
INSTANTIATE_BINARY("binary_sub_f16", half, SubOp)
INSTANTIATE_BINARY("binary_sub_bf16", bf16_t, SubOp)
INSTANTIATE_BINARY("binary_mul_f32", float, MulOp)
INSTANTIATE_BINARY("binary_mul_f16", half, MulOp)
INSTANTIATE_BINARY("binary_mul_bf16", bf16_t, MulOp)
INSTANTIATE_BINARY("binary_div_f32", float, DivOp)
INSTANTIATE_BINARY("binary_div_f16", half, DivOp)
INSTANTIATE_BINARY("binary_div_bf16", bf16_t, DivOp)
