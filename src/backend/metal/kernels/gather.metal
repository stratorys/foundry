struct GatherArgs {
  uint count;
  uint rows;
  uint cols;
  uint table_offset;
  uint table_row_stride;
  uint table_col_stride;
  StridedArgs indices;
};

template <typename T>
kernel void gather_rows(device const T *table [[buffer(0)]],
                        device const uint *indices [[buffer(1)]],
                        device T *output [[buffer(2)]],
                        constant GatherArgs &args [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= args.count) {
    return;
  }
  uint row = gid / args.cols;
  uint col = gid % args.cols;
  uint index = indices[strided_index(row, args.indices)];
  if (index >= args.rows) {
    output[gid] = from_float<T>(0.0f);
    return;
  }
  output[gid] = table[args.table_offset + index * args.table_row_stride +
                      col * args.table_col_stride];
}

#define INSTANTIATE_GATHER(name, T)                                            \
  template [[host_name(name)]] kernel void gather_rows<T>(                     \
      device const T *, device const uint *, device T *,                       \
      constant GatherArgs &, uint);

INSTANTIATE_GATHER("gather_f32", float)
INSTANTIATE_GATHER("gather_f16", half)
INSTANTIATE_GATHER("gather_bf16", bf16_t)
