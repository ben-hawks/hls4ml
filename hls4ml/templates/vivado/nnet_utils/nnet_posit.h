#ifndef NNET_POSIT_H_
#define NNET_POSIT_H_

#include "ap_fixed.h"
#include "ap_int.h"

#ifndef __SYNTHESIS__
#include <istream>
#endif

namespace nnet {

// Storage-only posit<N, ES>: holds the raw N-bit code. decode() returns the value as a fixed-point
// number wide enough to represent every posit<N, ES> value exactly, so arithmetic stays in fixed point.
template <int N, int ES> struct posit {
    static_assert(N >= 2 && N <= 32, "posit width must be between 2 and 32");
    static_assert(ES >= 0 && ES <= 4, "posit es must be between 0 and 4");

    static const int max_scale = (N - 2) << ES; // maxpos = 2^max_scale, minpos = 2^-max_scale
    typedef ap_fixed<2 * max_scale + 2, max_scale + 2> decode_t;

    ap_uint<N> bits;

    posit() : bits(0) {}
    posit(unsigned long long b) : bits(b) {}

    decode_t decode() const {
        #pragma HLS INLINE
        const ap_uint<N> nar = ap_uint<N>(1) << (N - 1);
        if (bits == 0 || bits == nar)
            return 0;

        const bool sign = bits[N - 1];
        const ap_uint<N> u = sign ? ap_uint<N>(~bits + 1) : bits;

        // Regime: run of identical bits starting below the sign bit
        const bool r0 = u[N - 2];
        int m = 0;
        bool run = true;
        for (int i = N - 2; i >= 0; i--) {
            #pragma HLS UNROLL
            if (run && u[i] == r0)
                m++;
            else
                run = false;
        }
        const int k = r0 ? m - 1 : -m;

        // Bits after the regime and its terminating bit: ES exponent bits (possibly truncated), then the fraction
        int rem = N - 2 - m;
        if (rem < 0)
            rem = 0;
        const unsigned tail = u.to_uint() & ((1u << rem) - 1);
        unsigned e;
        int nf;
        if (rem >= ES) {
            e = tail >> (rem - ES);
            nf = rem - ES;
        } else {
            e = tail << (ES - rem);
            nf = 0;
        }
        const unsigned sig = (1u << nf) | (tail & ((1u << nf) - 1));
        const int scale = k * (1 << ES) + int(e) - nf;

        decode_t v = sig;
        if (scale >= 0)
            v = v << scale;
        else
            v = v >> -scale;
        return sign ? decode_t(-v) : v;
    }
};

#ifndef __SYNTHESIS__
template <int N, int ES> std::istream &operator>>(std::istream &in, posit<N, ES> &p) {
    unsigned long long b;
    in >> b;
    p.bits = b;
    return in;
}
#endif

} // namespace nnet

#endif
