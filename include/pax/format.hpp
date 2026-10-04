#pragma once

#include "pax/common.hpp"
#include <array>
#include <vector>

namespace pax {

#pragma pack(push, 1)

// Container Header at offset 0x00
// [0x00 - 0x03] Magic Bytes: 0x50, 0x41, 0x58, 0x01 ("PAX\x01")
// [0x04 - 0x07] Flags & Engine Profile (Compression Level, Vectorization Flag)
// [0x08 - 0x0F] Uncompressed Total Size (64-bit uint)
// [0x10 - 0x17] Compressed Payload Size (64-bit uint)
// [0x18 - 0x1D] Reserved / padding to 0x1E
// [0x1E - 0x3D] Original File SHA-256 Checksum (32 bytes)
// [0x3E - 0x41] Number of Chunks (32-bit uint)
// [0x42 - ...]  Chunk Offset Table begins
struct ContainerHeader {
    uint8_t  magic[4];            // 0x00: "PAX\x01"
    uint32_t flags;               // 0x04: Engine profile & vectorization flags
    uint64_t uncompressed_size;   // 0x08: Total uncompressed bytes
    uint64_t compressed_payload;  // 0x10: Total compressed payload bytes
    uint8_t  reserved[6];         // 0x18: Padding to 0x1E
    uint8_t  sha256[32];          // 0x1E: Original SHA-256 digest
    uint32_t num_chunks;          // 0x3E: Total number of chunks
};
static_assert(sizeof(ContainerHeader) == 0x42, "ContainerHeader must be exactly 0x42 (66) bytes");

// Chunk Descriptor in the Chunk Table
struct ChunkDescriptor {
    uint32_t compressed_size;   // Compressed bytes in bitstream
    uint32_t uncompressed_size; // Original uncompressed bytes in chunk
    uint32_t checksum;          // CRC32 of uncompressed chunk
    uint8_t  method;            // ChunkMethod (0: STORED, 1: BWT+rANS, 2: ULTRA_MIXER)
    uint8_t  flags;             // Chunk flags (e.g. 0x01 = context seed warmed)
    uint16_t reserved;          // Reserved / alignment
    uint32_t extra;             // Transform parameter (e.g. BWT primary index)
};
static_assert(sizeof(ChunkDescriptor) == 20, "ChunkDescriptor must be 20 bytes");

#pragma pack(pop)

constexpr size_t HEADER_SIZE = sizeof(ContainerHeader); // 66 bytes (0x42)
constexpr size_t TAIL_SIZE   = 4;                       // 4 bytes: "XAP\xFF"
constexpr size_t SEED_TAIL_SIZE = 65536;                // 64 KB terminal context seed

} // namespace pax
