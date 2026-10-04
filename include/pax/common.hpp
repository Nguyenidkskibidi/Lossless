#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <span>
#include <string>
#include <memory>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace pax {

// Magic identifiers
inline constexpr uint8_t MAGIC_HEADER[4] = {0x50, 0x41, 0x58, 0x01}; // "PAX\x01"
inline constexpr uint8_t MAGIC_TAIL[4]   = {0x58, 0x41, 0x50, 0xFF}; // "XAP\xFF"

// Engine profile & flags
enum ProfileFlags : uint32_t {
    FLAG_PROFILE_FAST     = 1 << 0,
    FLAG_PROFILE_STANDARD = 1 << 1,
    FLAG_PROFILE_ULTRA    = 1 << 2,
    FLAG_VECTORIZED       = 1 << 3,
    FLAG_MULTITHREADED    = 1 << 4,
};

// Chunk methods
enum ChunkMethod : uint8_t {
    METHOD_STORED          = 0, // Uncompressed / fallback
    METHOD_BWT_RANS        = 1, // Transform: BWT + MTF/RLE + Interleaved rANS
    METHOD_ULTRA_MIXER     = 2, // Context Modeling (Order 1..32, Sparse) + Fixed-Point Logistic Mixer + SSE
    METHOD_TOKEN_BWT_RANS  = 3, // Transform: Tokenizer + BWT + MTF/RLE + Interleaved rANS
    METHOD_TOKEN_ULTRA     = 4, // Transform: Tokenizer + Ultra Context Mixer
};

// Architecture detection
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define PAX_HAS_NEON 1
#include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_CRC32)
#define PAX_HAS_ARM_CRC32 1
#include <arm_acle.h>
#endif

#if defined(__ARM_FEATURE_CRYPTO) || defined(__ARM_FEATURE_SHA2)
#define PAX_HAS_ARM_SHA2 1
#endif

#if defined(__AVX512F__)
#define PAX_HAS_AVX512 1
#include <immintrin.h>
#elif defined(__AVX2__)
#define PAX_HAS_AVX2 1
#include <immintrin.h>
#elif defined(__SSE4_2__)
#define PAX_HAS_SSE42 1
#include <nmmintrin.h>
#endif

// Endianness handling (All PAX containers are little-endian on disk)
inline uint32_t read_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
          (static_cast<uint32_t>(p[1]) << 8) |
          (static_cast<uint32_t>(p[2]) << 16) |
          (static_cast<uint32_t>(p[3]) << 24);
}

inline void write_le32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFF);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
}

inline uint64_t read_le64(const uint8_t* p) {
    return static_cast<uint64_t>(read_le32(p)) |
          (static_cast<uint64_t>(read_le32(p + 4)) << 32);
}

inline void write_le64(uint8_t* p, uint64_t v) {
    write_le32(p, static_cast<uint32_t>(v & 0xFFFFFFFF));
    write_le32(p + 4, static_cast<uint32_t>((v >> 32) & 0xFFFFFFFF));
}

// Memory alignment helper
template <typename T>
inline T align_up(T val, T alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

} // namespace pax
