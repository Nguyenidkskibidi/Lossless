#pragma once

#include "pax/common.hpp"
#include <vector>

namespace pax {

class MTF {
public:
    // Move-to-Front (MTF) forward transform
    static std::vector<uint8_t> encode(const uint8_t* in, size_t n);

    // Inverted Move-to-Front (IMTF) reverse transform
    static std::vector<uint8_t> decode(const uint8_t* in, size_t n);

    // RLE Stage 1: Collapses long runs (>= 4 identical symbols) into 4 bytes + count
    static std::vector<uint8_t> rle1_encode(const uint8_t* in, size_t n);
    static std::vector<uint8_t> rle1_decode(const uint8_t* in, size_t n);

    // Zero-Run RLE: Collapses runs of zeroes post-MTF
    static std::vector<uint8_t> zero_rle_encode(const uint8_t* in, size_t n);
    static std::vector<uint8_t> zero_rle_decode(const uint8_t* in, size_t n);
};

} // namespace pax
