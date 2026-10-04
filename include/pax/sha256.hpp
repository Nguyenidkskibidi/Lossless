#pragma once

#include "pax/common.hpp"
#include <string>
#include <array>

namespace pax {

class SHA256 {
public:
    static constexpr size_t DIGEST_SIZE = 32;
    static constexpr size_t BLOCK_SIZE  = 64;

    SHA256();
    void reset();
    void update(const void* data, size_t length);
    void finalize(uint8_t digest[DIGEST_SIZE]);
    std::array<uint8_t, DIGEST_SIZE> finalize();

    static std::array<uint8_t, DIGEST_SIZE> hash(const void* data, size_t length);
    static std::string to_hex(const uint8_t digest[DIGEST_SIZE]);
    static std::string to_hex(const std::array<uint8_t, DIGEST_SIZE>& digest);

private:
    void transform(const uint8_t block[BLOCK_SIZE]);

    uint32_t state_[8];
    uint64_t count_;
    uint8_t buffer_[BLOCK_SIZE];
};

} // namespace pax
