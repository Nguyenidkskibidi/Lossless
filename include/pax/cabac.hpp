#pragma once

#include "pax/common.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace pax {

class CabacMtf {
public:
    // Compress MTF stream using Context-Adaptive Binary Arithmetic Coding
    static std::vector<uint8_t> compress(const uint8_t* in, size_t n);

    // Decompress CABAC MTF payload back to bit-exact MTF stream
    static std::vector<uint8_t> decompress(const uint8_t* in, size_t comp_len);
};

} // namespace pax
