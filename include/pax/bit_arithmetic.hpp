#pragma once

#include "pax/common.hpp"
#include <vector>

namespace pax {

// Bit-level Range Arithmetic Coder
class RangeBitEncoder {
public:
    RangeBitEncoder();
    void encode_bit(int bit, uint16_t prob1);
    void finish();
    const std::vector<uint8_t>& get_buffer() const { return out_; }
    std::vector<uint8_t> release_buffer() { return std::move(out_); }

private:
    uint32_t low_;
    uint32_t range_;
    std::vector<uint8_t> out_;
};

class RangeBitDecoder {
public:
    RangeBitDecoder(const uint8_t* data, size_t length);
    int decode_bit(uint16_t prob1);

private:
    uint32_t low_;
    uint32_t code_;
    uint32_t range_;
    const uint8_t* in_ptr_;
    const uint8_t* in_end_;
};

} // namespace pax
