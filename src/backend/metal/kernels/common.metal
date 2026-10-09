#include <metal_stdlib>

using namespace metal;

struct StridedArgs {
  uint count;
  uint rank;
  uint dims[4];
  uint strides[4];
  uint offset;
};

inline uint strided_index(uint linear, constant StridedArgs &args) {
  uint index = args.offset;
  for (uint axis = args.rank; axis > 0; --axis) {
    uint dim = args.dims[axis - 1];
    index += (linear % dim) * args.strides[axis - 1];
    linear /= dim;
  }
  return index;
}

struct bf16_t {
  ushort bits;
};

inline float to_float(float value) { return value; }

inline float to_float(half value) { return float(value); }

inline float to_float(bf16_t value) {
  return as_type<float>(uint(value.bits) << 16);
}

template <typename T> T from_float(float value);

template <> inline float from_float<float>(float value) { return value; }

template <> inline half from_float<half>(float value) { return half(value); }

template <> inline bf16_t from_float<bf16_t>(float value) {
  uint bits = as_type<uint>(value);
  if (isnan(value)) {
    return bf16_t{ushort((bits >> 16) | 0x0040u)};
  }
  return bf16_t{ushort((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16)};
}
