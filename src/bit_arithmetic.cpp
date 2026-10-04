#include "pax/bit_arithmetic.hpp"

namespace pax {

constexpr uint32_t TOP_MASK = 1 << 24;
constexpr uint32_t BOT_MASK = 1 << 16;
constexpr uint32_t PROB_SCALE = 4096;

RangeBitEncoder::RangeBitEncoder() : low_(0), range_(0xFFFFFFFF) {
    out_.reserve(4096);
}

void RangeBitEncoder::encode_bit(int bit, uint16_t prob1) {
    uint32_t r0 = static_cast<uint32_t>((static_cast<uint64_t>(range_) * (PROB_SCALE - prob1)) >> 12);
    if (bit == 0) {
        range_ = r0;
    } else {
        low_ += r0;
        range_ -= r0;
    }

    while ((low_ ^ (low_ + range_)) < TOP_MASK ||
           (range_ < BOT_MASK && ((range_ = -low_ & (BOT_MASK - 1)), true))) {
        out_.push_back(static_cast<uint8_t>(low_ >> 24));
        low_ <<= 8;
        range_ <<= 8;
    }
}

void RangeBitEncoder::finish() {
    for (int i = 0; i < 4; ++i) {
        out_.push_back(static_cast<uint8_t>(low_ >> 24));
        low_ <<= 8;
    }
}

RangeBitDecoder::RangeBitDecoder(const uint8_t* data, size_t length)
    : low_(0), code_(0), range_(0xFFFFFFFF), in_ptr_(data), in_end_(data + length) {
    for (int i = 0; i < 4; ++i) {
        code_ = (code_ << 8) | (in_ptr_ < in_end_ ? *in_ptr_++ : 0);
    }
}

int RangeBitDecoder::decode_bit(uint16_t prob1) {
    uint32_t r0 = static_cast<uint32_t>((static_cast<uint64_t>(range_) * (PROB_SCALE - prob1)) >> 12);
    int bit;
    if (code_ - low_ < r0) {
        bit = 0;
        range_ = r0;
    } else {
        bit = 1;
        low_ += r0;
        range_ -= r0;
    }

    while ((low_ ^ (low_ + range_)) < TOP_MASK ||
           (range_ < BOT_MASK && ((range_ = -low_ & (BOT_MASK - 1)), true))) {
        code_ = (code_ << 8) | (in_ptr_ < in_end_ ? *in_ptr_++ : 0);
        low_ <<= 8;
        range_ <<= 8;
    }
    return bit;
}

} // namespace pax
