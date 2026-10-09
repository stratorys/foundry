#define MATMUL_TILE 16

struct MatmulOperand {
  uint offset;
  uint row_stride;
  uint col_stride;
  uint batch_strides[4];
};

struct MatmulArgs {
  uint m;
  uint n;
  uint k;
  uint batch_rank;
  uint batch_dims[4];
  MatmulOperand lhs;
  MatmulOperand rhs;
};

inline uint matmul_base(uint batch, constant MatmulArgs &args,
                        constant MatmulOperand &operand) {
  uint index = operand.offset;
  for (uint axis = args.batch_rank; axis > 0; --axis) {
    uint dim = args.batch_dims[axis - 1];
    index += (batch % dim) * operand.batch_strides[axis - 1];
    batch /= dim;
  }
  return index;
}

template <typename T>
kernel void matmul_tiled(device const T *lhs [[buffer(0)]],
                         device const T *rhs [[buffer(1)]],
                         device T *output [[buffer(2)]],
                         constant MatmulArgs &args [[buffer(3)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 tid [[thread_position_in_threadgroup]]) {
  threadgroup float lhs_tile[MATMUL_TILE][MATMUL_TILE];
  threadgroup float rhs_tile[MATMUL_TILE][MATMUL_TILE];
  uint row = group.y * MATMUL_TILE + tid.y;
  uint col = group.x * MATMUL_TILE + tid.x;
  uint lhs_base = matmul_base(group.z, args, args.lhs);
  uint rhs_base = matmul_base(group.z, args, args.rhs);
  float acc = 0.0f;
  for (uint k_start = 0; k_start < args.k; k_start += MATMUL_TILE) {
    uint lhs_k = k_start + tid.x;
    uint rhs_k = k_start + tid.y;
    lhs_tile[tid.y][tid.x] =
        (row < args.m && lhs_k < args.k)
            ? to_float(lhs[lhs_base + row * args.lhs.row_stride +
                           lhs_k * args.lhs.col_stride])
            : 0.0f;
    rhs_tile[tid.y][tid.x] =
        (rhs_k < args.k && col < args.n)
            ? to_float(rhs[rhs_base + rhs_k * args.rhs.row_stride +
                           col * args.rhs.col_stride])
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint inner = 0; inner < MATMUL_TILE; ++inner) {
      acc += lhs_tile[tid.y][inner] * rhs_tile[inner][tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < args.m && col < args.n) {
    output[(group.z * args.m + row) * args.n + col] = from_float<T>(acc);
  }
}

#define INSTANTIATE_MATMUL(name, T)                                            \
  template [[host_name(name)]] kernel void matmul_tiled<T>(                    \
      device const T *, device const T *, device T *, constant MatmulArgs &,   \
      uint3, uint3);

INSTANTIATE_MATMUL("matmul_f32", float)
INSTANTIATE_MATMUL("matmul_f16", half)
INSTANTIATE_MATMUL("matmul_bf16", bf16_t)
