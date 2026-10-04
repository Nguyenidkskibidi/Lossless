#pragma once

#include "pax/common.hpp"
#include "pax/format.hpp"
#include "pax/sha256.hpp"
#include "pax/crc32.hpp"
#include <string>
#include <vector>

namespace pax {

struct CompressionOptions {
    bool ultra = false;
    int threads = 0; // 0 = hardware concurrency
    size_t chunk_size = 16 * 1024 * 1024; // 16 MB default for PAX v2.0
    bool chunk_size_explicit = false;
    bool verbose = false;
};

struct VerificationResult {
    bool success = false;
    uint64_t uncompressed_size = 0;
    uint64_t compressed_size = 0;
    uint32_t num_chunks = 0;
    std::string expected_sha256;
    std::string computed_sha256;
    std::string error_message;
};

class PaxEngine {
public:
    // Compress input file to .pax container
    static bool compress_file(const std::string& input_path,
                             const std::string& output_path,
                             const CompressionOptions& options);

    // Decompress .pax container to restored file
    static bool decompress_file(const std::string& input_path,
                               const std::string& output_path,
                               bool verbose = false);

    // Verify .pax container integrity (SHA-256 & CRC-32) without writing to disk
    static VerificationResult verify_file(const std::string& pax_path, bool verbose = false);

    // In-memory compression and decompression (useful for testing & benchmarks)
    static std::vector<uint8_t> compress_memory(const uint8_t* in,
                                               size_t n,
                                               const CompressionOptions& options);

    static std::vector<uint8_t> decompress_memory(const uint8_t* in,
                                                 size_t comp_len);
};

} // namespace pax
