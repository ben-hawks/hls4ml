#ifndef NNET_LUT_WEIGHT_H_
#define NNET_LUT_WEIGHT_H_

#include "ap_int.h"

#ifndef __SYNTHESIS__
#include <istream>
#endif

namespace nnet {

// Weight stored as an N-bit code; decode() reads its fixed-point value from table_T::lookup(), a per-type table
// written by hls4ml (a fitted codebook).
template <int N, class decode_T, class table_T> struct lut_weight {
    typedef decode_T decode_t;

    ap_uint<N> bits;

    lut_weight() : bits(0) {}
    lut_weight(unsigned long long b) : bits(b) {}

    decode_t decode() const {
        #pragma HLS INLINE
        return table_T::lookup(bits.to_uint());
    }
};

#ifndef __SYNTHESIS__
template <int N, class decode_T, class table_T>
std::istream &operator>>(std::istream &in, lut_weight<N, decode_T, table_T> &w) {
    unsigned long long b;
    in >> b;
    w.bits = b;
    return in;
}
#endif

} // namespace nnet

#endif
