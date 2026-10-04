#pragma once

#include "pax/common.hpp"
#include <vector>

namespace pax {

class RANS4 {
public:
    static constexpr uint32_t SCALE_BITS = 12;
    static constexpr uint32_t SCALE_M    = 1 << SCALE_BITS; // 4096
    static constexpr uint32_t RANS_L     = 1 << 16;         // 65536

    struct FrequencyTable {
        uint16_t freq[256];
        uint16_t cum[257];
        uint8_t  lut[SCALE_M];

        void build(const uint32_t counts[256], size_t total_count);
        void serialize(std::vector<uint8_t>& out) const;
        bool deserialize(const uint8_t*& in_ptr, const uint8_t* in_end);
    };

    // Compress input using 4-way interleaved rANS
    static std::vector<uint8_t> compress(const uint8_t* in, size_t n);

    // Decompress payload to restored buffer
    static std::vector<uint8_t> decompress(const uint8_t* in, size_t comp_len);
};

} // namespace pax
