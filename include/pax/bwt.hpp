#pragma once

#include "pax/common.hpp"
#include <vector>
#include <utility>

namespace pax {

class BWT {
public:
    // Forward BWT transform
    // Returns transformed bytes and the primary index
    static std::pair<std::vector<uint8_t>, uint32_t> encode(const uint8_t* in, size_t n, int num_threads = 1);

    // Inverse BWT transform
    // Reconstructs original bytes from BWT output and primary index
    static std::vector<uint8_t> decode(const uint8_t* in, size_t n, uint32_t primary_index);

};

} // namespace pax
