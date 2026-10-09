struct NegOp {
  static float apply(float x) { return -x; }
};

struct ExpOp {
  static float apply(float x) { return exp(x); }
};

struct SqrtOp {
  static float apply(float x) { return sqrt(x); }
};

struct RecipOp {
  static float apply(float x) { return 1.0f / x; }
};

template <typename T, typename Op>
kernel void unary_strided(device const T *input [[buffer(0)]],
                          device T *output [[buffer(1)]],
                          constant StridedArgs &args [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid >= args.count) {
    return;
  }
  output[gid] =
      from_float<T>(Op::apply(to_float(input[strided_index(gid, args)])));
}

#define INSTANTIATE_UNARY(name, T, Op)                                         \
  template [[host_name(name)]] kernel void unary_strided<T, Op>(               \
      device const T *, device T *, constant StridedArgs &, uint);

INSTANTIATE_UNARY("unary_neg_f32", float, NegOp)
INSTANTIATE_UNARY("unary_neg_f16", half, NegOp)
INSTANTIATE_UNARY("unary_neg_bf16", bf16_t, NegOp)
INSTANTIATE_UNARY("unary_exp_f32", float, ExpOp)
INSTANTIATE_UNARY("unary_exp_f16", half, ExpOp)
INSTANTIATE_UNARY("unary_exp_bf16", bf16_t, ExpOp)
INSTANTIATE_UNARY("unary_sqrt_f32", float, SqrtOp)
INSTANTIATE_UNARY("unary_sqrt_f16", half, SqrtOp)
INSTANTIATE_UNARY("unary_sqrt_bf16", bf16_t, SqrtOp)
INSTANTIATE_UNARY("unary_recip_f32", float, RecipOp)
INSTANTIATE_UNARY("unary_recip_f16", half, RecipOp)
INSTANTIATE_UNARY("unary_recip_bf16", bf16_t, RecipOp)
