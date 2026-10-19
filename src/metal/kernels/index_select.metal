struct IndexSelectArgs {
  uint count;
  uint rows;
  uint cols;
  uint table_offset;
  uint table_row_stride;
  uint table_col_stride;
  StridedArgs indices;
};

template <typename T>
kernel void index_select_rows(device const T *table [[buffer(0)]],
                              device const uint *indices [[buffer(1)]],
                              device T *output [[buffer(2)]],
                              constant IndexSelectArgs &args [[buffer(3)]],
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

#define INSTANTIATE_INDEX_SELECT(name, T)                                      \
  template [[host_name(name)]] kernel void index_select_rows<T>(               \
      device const T *, device const uint *, device T *,                       \
      constant IndexSelectArgs &, uint);

INSTANTIATE_INDEX_SELECT("index_select_f32", float)
INSTANTIATE_INDEX_SELECT("index_select_bf16", bf16_t)
