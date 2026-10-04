#include "pax/engine.hpp"
#include "pax/tokenizer.hpp"
#include "pax/bwt.hpp"
#include "pax/mtf.hpp"
#include "pax/rans.hpp"
#include "pax/neural_mixer.hpp"
#include "pax/thread_pool.hpp"
#include "pax/io.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace pax {

std::vector<uint8_t> PaxEngine::compress_memory(const uint8_t* in,
                                               size_t n,
                                               const CompressionOptions& options) {
    // 1. Calculate overall SHA-256 of original raw input
    auto original_sha = SHA256::hash(in, n);

    size_t chunk_size = options.chunk_size;
    if (options.ultra && !options.chunk_size_explicit) {
        chunk_size = std::max<size_t>(65536, n);
    } else {
        chunk_size = std::max<size_t>(65536, options.chunk_size);
    }
    uint32_t num_chunks = static_cast<uint32_t>((n + chunk_size - 1) / chunk_size);
    if (n == 0) num_chunks = 0;

    int num_threads = options.threads > 0 ? options.threads : static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads < 1) num_threads = 1;

    struct ProcessedChunk {
        ChunkDescriptor desc{};
        std::vector<uint8_t> payload;
    };

    std::vector<ProcessedChunk> processed(num_chunks);
    std::atomic<uint32_t> done_chunks(0);
    std::atomic<uint64_t> done_raw_bytes(0);
    std::atomic<uint64_t> done_comp_bytes(0);
    auto t0 = std::chrono::high_resolution_clock::now();

    auto compress_single_chunk = [&](size_t i) {
        size_t c_start = i * chunk_size;
        size_t c_len = std::min(chunk_size, n - c_start);
        const uint8_t* c_data = in + c_start;

        uint32_t crc = CRC32::calculate(c_data, c_len);

        const uint8_t* seed_ptr = nullptr;
        size_t seed_len = 0;
        if (i > 0) {
            seed_len = std::min(SEED_TAIL_SIZE, c_start);
            seed_ptr = in + c_start - seed_len;
        }

        std::vector<uint8_t> comp_payload;
        uint8_t method = METHOD_STORED;
        uint32_t extra = 0;

        if (options.ultra) {
            auto progress_cb = [&](size_t done, size_t total) {
                if (num_chunks == 1) {
                    auto now = std::chrono::high_resolution_clock::now();
                    double el = std::chrono::duration<double>(now - t0).count();
                    double mb_s = el > 0 ? (done / (1024.0 * 1024.0)) / el : 0.0;
                    double pct = (100.0 * done) / total;
                    double eta = (mb_s > 0) ? (static_cast<double>(total - done) / (1024.0 * 1024.0)) / mb_s : 0.0;
                    int eta_m = static_cast<int>(eta) / 60;
                    int eta_s = static_cast<int>(eta) % 60;
                    std::cout << "\r[PAX Ultra Progress] " << (done / (1024 * 1024)) << " MB / "
                              << (total / (1024 * 1024)) << " MB (" << std::fixed << std::setprecision(1) << pct
                              << "%) - " << std::setprecision(2) << mb_s << " MB/s - ETA: "
                              << eta_m << "m " << (eta_s < 10 ? "0" : "") << eta_s << "s" << std::flush;
                }
            };

            auto tokenized = Tokenizer::transform(c_data, c_len);
            if (!tokenized.empty()) {
                comp_payload = ContextMixer::compress(tokenized.data(), tokenized.size(), seed_ptr, seed_len, progress_cb);
                method = METHOD_TOKEN_ULTRA;
                extra = static_cast<uint32_t>(tokenized.size());
            } else {
                comp_payload = ContextMixer::compress(c_data, c_len, seed_ptr, seed_len, progress_cb);
                method = METHOD_ULTRA_MIXER;
                extra = 0;
            }
            if (num_chunks == 1) {
                std::cout << "\n";
            }
        } else {
            // Stage 0: Dynamic Tokenizer Pre-transform for natural text / XML
            auto tokenized = Tokenizer::transform(c_data, c_len);
            if (!tokenized.empty()) {
                // Stage 1: BWT on tokenized stream
                auto [bwt_data, primary] = BWT::encode(tokenized.data(), tokenized.size(), num_threads);
                // Stage 2: MTF
                auto mtf_data = MTF::encode(bwt_data.data(), bwt_data.size());
                // Stage 3: Zero-RLE
                auto rle_data = MTF::zero_rle_encode(mtf_data.data(), mtf_data.size());
                // Stage 4: 4-way Interleaved rANS
                comp_payload = RANS4::compress(rle_data.data(), rle_data.size());
                method = METHOD_TOKEN_BWT_RANS;
                extra = primary;
            } else {
                // Fallback: Standard BWT on raw stream
                auto [bwt_data, primary] = BWT::encode(c_data, c_len, num_threads);
                auto mtf_data = MTF::encode(bwt_data.data(), bwt_data.size());
                auto rle_data = MTF::zero_rle_encode(mtf_data.data(), mtf_data.size());
                comp_payload = RANS4::compress(rle_data.data(), rle_data.size());
                method = METHOD_BWT_RANS;
                extra = primary;
            }
        }

        // If compressed data expands, fallback to raw stored
        if (comp_payload.size() >= c_len) {
            comp_payload.assign(c_data, c_data + c_len);
            method = METHOD_STORED;
            extra = 0;
        }

        processed[i].desc.compressed_size   = static_cast<uint32_t>(comp_payload.size());
        processed[i].desc.uncompressed_size = static_cast<uint32_t>(c_len);
        processed[i].desc.checksum          = crc;
        processed[i].desc.method            = method;
        processed[i].desc.flags             = (seed_ptr != nullptr ? 1 : 0);
        processed[i].desc.reserved          = 0;
        processed[i].desc.extra             = extra;
        processed[i].payload                = std::move(comp_payload);

        uint32_t completed = ++done_chunks;
        uint64_t raw_done = (done_raw_bytes += c_len);
        uint64_t comp_done = (done_comp_bytes += processed[i].desc.compressed_size);

        if (num_chunks > 5 && (completed % std::max<uint32_t>(1, num_chunks / 10) == 0 || completed == num_chunks)) {
            auto now = std::chrono::high_resolution_clock::now();
            double el = std::chrono::duration<double>(now - t0).count();
            double mb_s = el > 0 ? (raw_done / (1024.0 * 1024.0)) / el : 0.0;
            double pct = (100.0 * completed) / num_chunks;
            double ratio = raw_done > 0 ? (100.0 * comp_done) / raw_done : 0.0;
            std::cout << "[PAX Progress] Chunks: " << completed << "/" << num_chunks
                      << " (" << std::fixed << std::setprecision(1) << pct << "%) - "
                      << mb_s << " MB/s - Ratio: " << ratio << "%\n" << std::flush;
        }
    };

    if (num_chunks > 1 && num_threads > 1) {
        ThreadPool pool(num_threads);
        std::vector<std::future<void>> futures;
        for (size_t i = 0; i < num_chunks; ++i) {
            futures.push_back(pool.enqueue(compress_single_chunk, i));
        }
        for (auto& f : futures) f.get();
    } else {
        for (size_t i = 0; i < num_chunks; ++i) {
            compress_single_chunk(i);
        }
    }

    // 2. Calculate total payload size
    uint64_t total_payload_size = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        total_payload_size += processed[i].payload.size();
    }

    // 3. Assemble binary container
    ContainerHeader header{};
    std::memcpy(header.magic, MAGIC_HEADER, 4);
    header.flags = options.ultra ? FLAG_PROFILE_ULTRA : FLAG_PROFILE_STANDARD;
#if defined(PAX_HAS_NEON) || defined(PAX_HAS_AVX512) || defined(PAX_HAS_AVX2)
    header.flags |= FLAG_VECTORIZED;
#endif
    if (num_threads > 1) {
        header.flags |= FLAG_MULTITHREADED;
    }
    header.uncompressed_size  = static_cast<uint64_t>(n);
    header.compressed_payload = total_payload_size;
    std::memset(header.reserved, 0, sizeof(header.reserved));
    std::memcpy(header.sha256, original_sha.data(), 32);
    header.num_chunks = num_chunks;

    size_t total_container_size = sizeof(ContainerHeader) +
                                  num_chunks * sizeof(ChunkDescriptor) +
                                  static_cast<size_t>(total_payload_size) +
                                  sizeof(MAGIC_TAIL);

    std::vector<uint8_t> out;
    out.reserve(total_container_size);

    // Write Header (0x42 bytes)
    const uint8_t* h_ptr = reinterpret_cast<const uint8_t*>(&header);
    out.insert(out.end(), h_ptr, h_ptr + sizeof(ContainerHeader));

    // Write Chunk Table
    for (size_t i = 0; i < num_chunks; ++i) {
        const uint8_t* d_ptr = reinterpret_cast<const uint8_t*>(&processed[i].desc);
        out.insert(out.end(), d_ptr, d_ptr + sizeof(ChunkDescriptor));
    }

    // Write Payloads
    for (size_t i = 0; i < num_chunks; ++i) {
        out.insert(out.end(), processed[i].payload.begin(), processed[i].payload.end());
    }

    // Write Tail Sentinel ("XAP\xFF")
    out.insert(out.end(), MAGIC_TAIL, MAGIC_TAIL + 4);

    return out;
}

std::vector<uint8_t> PaxEngine::decompress_memory(const uint8_t* in, size_t comp_len) {
    if (comp_len < sizeof(ContainerHeader) + sizeof(MAGIC_TAIL)) {
        throw std::runtime_error("Corrupted .pax: File too small for header");
    }

    // Verify header magic
    if (std::memcmp(in, MAGIC_HEADER, 4) != 0) {
        throw std::runtime_error("Invalid .pax magic bytes in header");
    }

    // Verify tail sentinel
    if (std::memcmp(in + comp_len - 4, MAGIC_TAIL, 4) != 0) {
        throw std::runtime_error("Invalid .pax tail sentinel (file may be truncated)");
    }

    const ContainerHeader* header = reinterpret_cast<const ContainerHeader*>(in);
    uint32_t num_chunks = header->num_chunks;
    uint64_t expected_total_size = header->uncompressed_size;

    size_t table_offset = sizeof(ContainerHeader);
    size_t table_size = num_chunks * sizeof(ChunkDescriptor);
    size_t payload_offset = table_offset + table_size;

    if (payload_offset + 4 > comp_len) {
        throw std::runtime_error("Corrupted .pax: Truncated chunk table");
    }

    std::vector<ChunkDescriptor> chunk_descs(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        std::memcpy(&chunk_descs[i], in + table_offset + i * sizeof(ChunkDescriptor), sizeof(ChunkDescriptor));
    }

    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(expected_total_size));

    size_t curr_payload_offset = payload_offset;

    for (size_t i = 0; i < num_chunks; ++i) {
        const auto& desc = chunk_descs[i];
        if (curr_payload_offset + desc.compressed_size + 4 > comp_len) {
            throw std::runtime_error("Corrupted .pax: Chunk payload exceeds container boundary");
        }

        const uint8_t* payload_ptr = in + curr_payload_offset;
        std::vector<uint8_t> chunk_out;

        if (desc.method == METHOD_STORED) {
            chunk_out.assign(payload_ptr, payload_ptr + desc.compressed_size);
        } else if (desc.method == METHOD_BWT_RANS) {
            // Stage 1: Interleaved rANS decode
            auto rle_data = RANS4::decompress(payload_ptr, desc.compressed_size);
            // Stage 2: Zero-RLE decode
            auto mtf_data = MTF::zero_rle_decode(rle_data.data(), rle_data.size());
            // Stage 3: MTF decode
            auto bwt_data = MTF::decode(mtf_data.data(), mtf_data.size());
            // Stage 4: Inverse BWT
            chunk_out = BWT::decode(bwt_data.data(), bwt_data.size(), desc.extra);
        } else if (desc.method == METHOD_TOKEN_BWT_RANS) {
            // Stage 1: Interleaved rANS decode
            auto rle_data = RANS4::decompress(payload_ptr, desc.compressed_size);
            // Stage 2: Zero-RLE decode
            auto mtf_data = MTF::zero_rle_decode(rle_data.data(), rle_data.size());
            // Stage 3: MTF decode
            auto bwt_data = MTF::decode(mtf_data.data(), mtf_data.size());
            // Stage 4: Inverse BWT
            auto tokenized = BWT::decode(bwt_data.data(), bwt_data.size(), desc.extra);
            // Stage 5: Inverse Tokenizer transform
            chunk_out = Tokenizer::inverse_transform(tokenized.data(), tokenized.size());
        } else if (desc.method == METHOD_ULTRA_MIXER) {
            const uint8_t* seed_ptr = nullptr;
            size_t seed_len = 0;
            if (i > 0 && (desc.flags & 1) && !out.empty()) {
                seed_len = std::min<size_t>(SEED_TAIL_SIZE, out.size());
                seed_ptr = out.data() + out.size() - seed_len;
            }
            chunk_out = ContextMixer::decompress(payload_ptr, desc.compressed_size, desc.uncompressed_size, seed_ptr, seed_len);
        } else if (desc.method == METHOD_TOKEN_ULTRA) {
            const uint8_t* seed_ptr = nullptr;
            size_t seed_len = 0;
            if (i > 0 && (desc.flags & 1) && !out.empty()) {
                seed_len = std::min<size_t>(SEED_TAIL_SIZE, out.size());
                seed_ptr = out.data() + out.size() - seed_len;
            }
            auto tokenized = ContextMixer::decompress(payload_ptr, desc.compressed_size, desc.extra, seed_ptr, seed_len);
            chunk_out = Tokenizer::inverse_transform(tokenized.data(), tokenized.size());
        } else {
            throw std::runtime_error("Unsupported chunk compression method: " + std::to_string(desc.method));
        }

        if (chunk_out.size() != desc.uncompressed_size) {
            throw std::runtime_error("Chunk uncompressed size mismatch at chunk " + std::to_string(i));
        }

        // Verify Chunk CRC32
        uint32_t chunk_crc = CRC32::calculate(chunk_out.data(), chunk_out.size());
        if (chunk_crc != desc.checksum) {
            throw std::runtime_error("Chunk CRC32 checksum mismatch at chunk " + std::to_string(i));
        }

        out.insert(out.end(), chunk_out.begin(), chunk_out.end());
        curr_payload_offset += desc.compressed_size;
    }

    if (out.size() != expected_total_size) {
        throw std::runtime_error("Final uncompressed size mismatch");
    }

    // Verify Overall SHA-256 Checksum
    auto computed_sha = SHA256::hash(out.data(), out.size());
    if (std::memcmp(computed_sha.data(), header->sha256, 32) != 0) {
        throw std::runtime_error("SHA-256 integrity verification failed: container is corrupted or tampered");
    }

    return out;
}

bool PaxEngine::compress_file(const std::string& input_path,
                             const std::string& output_path,
                             const CompressionOptions& options) {
    auto t0 = std::chrono::high_resolution_clock::now();

    MappedFileView view(input_path);
    size_t in_size = view.size();

    std::cout << "[PAX] Compressing: " << input_path << " (" << in_size << " bytes = "
              << std::fixed << std::setprecision(2) << (in_size / (1024.0 * 1024.0)) << " MB)\n";
    std::cout << "[PAX] Profile: " << (options.ultra ? "ULTRA v3.0 (Tokenizer + 64MB Long Match + Mahoney StateTable 14-Model Mixer + Range Coder)" : "STANDARD (Tokenizer + pBWT + MTF + Interleaved rANS)") << "\n";
    if (options.ultra && !options.chunk_size_explicit) {
        std::cout << "[PAX] Mode: Continuous Streaming (Global Adaptive Model)\n";
    } else {
        std::cout << "[PAX] Chunk size: " << (options.chunk_size / (1024 * 1024)) << " MB\n";
    }

    auto comp_data = compress_memory(view.data(), in_size, options);

    write_file(output_path, comp_data.data(), comp_data.size());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double mb = static_cast<double>(in_size) / (1024.0 * 1024.0);
    double mb_s = elapsed_sec > 0.000001 ? (mb / elapsed_sec) : 0.0;
    double ratio = in_size > 0 ? (100.0 * comp_data.size() / in_size) : 0.0;

    std::cout << "[PAX] Compressed: " << in_size << " -> " << comp_data.size()
              << " bytes (" << std::fixed << std::setprecision(2) << ratio << "%) in "
              << elapsed_sec << "s (" << mb_s << " MB/s)\n";

    return true;
}

bool PaxEngine::decompress_file(const std::string& input_path,
                               const std::string& output_path,
                               bool verbose) {
    auto t0 = std::chrono::high_resolution_clock::now();

    MappedFileView view(input_path);
    size_t comp_len = view.size();
    const uint8_t* in = view.data();

    if (comp_len < sizeof(ContainerHeader) + sizeof(MAGIC_TAIL)) {
        throw std::runtime_error("Corrupted .pax: File too small for header");
    }

    if (std::memcmp(in, MAGIC_HEADER, 4) != 0) {
        throw std::runtime_error("Invalid .pax magic bytes in header");
    }

    if (std::memcmp(in + comp_len - 4, MAGIC_TAIL, 4) != 0) {
        throw std::runtime_error("Invalid .pax tail sentinel (file may be truncated)");
    }

    const ContainerHeader* header = reinterpret_cast<const ContainerHeader*>(in);
    uint32_t num_chunks = header->num_chunks;
    uint64_t expected_total_size = header->uncompressed_size;

    size_t table_offset = sizeof(ContainerHeader);
    size_t table_size = num_chunks * sizeof(ChunkDescriptor);
    size_t payload_offset = table_offset + table_size;

    if (payload_offset + 4 > comp_len) {
        throw std::runtime_error("Corrupted .pax: Truncated chunk table");
    }

    std::vector<ChunkDescriptor> chunk_descs(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        std::memcpy(&chunk_descs[i], in + table_offset + i * sizeof(ChunkDescriptor), sizeof(ChunkDescriptor));
    }

    std::cout << "[PAX] Decompressing: " << input_path << " (" << comp_len << " bytes, "
              << num_chunks << " chunks, expected " << (expected_total_size / (1024.0 * 1024.0)) << " MB)...\n";

    std::ofstream out_file(output_path, std::ios::binary | std::ios::trunc);
    if (!out_file) {
        throw std::runtime_error("Failed to open output file for writing: " + output_path);
    }

    SHA256 running_sha;
    size_t curr_payload_offset = payload_offset;
    std::vector<uint8_t> seed_tail;
    uint64_t total_restored = 0;

    for (size_t i = 0; i < num_chunks; ++i) {
        const auto& desc = chunk_descs[i];
        if (curr_payload_offset + desc.compressed_size + 4 > comp_len) {
            throw std::runtime_error("Corrupted .pax: Chunk payload exceeds container boundary");
        }

        const uint8_t* payload_ptr = in + curr_payload_offset;
        std::vector<uint8_t> chunk_out;

        if (desc.method == METHOD_STORED) {
            chunk_out.assign(payload_ptr, payload_ptr + desc.compressed_size);
        } else if (desc.method == METHOD_BWT_RANS) {
            auto rle_data = RANS4::decompress(payload_ptr, desc.compressed_size);
            auto mtf_data = MTF::zero_rle_decode(rle_data.data(), rle_data.size());
            auto bwt_data = MTF::decode(mtf_data.data(), mtf_data.size());
            chunk_out = BWT::decode(bwt_data.data(), bwt_data.size(), desc.extra);
        } else if (desc.method == METHOD_TOKEN_BWT_RANS) {
            auto rle_data = RANS4::decompress(payload_ptr, desc.compressed_size);
            auto mtf_data = MTF::zero_rle_decode(rle_data.data(), rle_data.size());
            auto bwt_data = MTF::decode(mtf_data.data(), mtf_data.size());
            auto tokenized = BWT::decode(bwt_data.data(), bwt_data.size(), desc.extra);
            chunk_out = Tokenizer::inverse_transform(tokenized.data(), tokenized.size());
        } else if (desc.method == METHOD_ULTRA_MIXER) {
            const uint8_t* seed_ptr = seed_tail.empty() ? nullptr : seed_tail.data();
            size_t seed_len = seed_tail.size();
            chunk_out = ContextMixer::decompress(payload_ptr, desc.compressed_size, desc.uncompressed_size, seed_ptr, seed_len);
        } else if (desc.method == METHOD_TOKEN_ULTRA) {
            const uint8_t* seed_ptr = seed_tail.empty() ? nullptr : seed_tail.data();
            size_t seed_len = seed_tail.size();
            auto tokenized = ContextMixer::decompress(payload_ptr, desc.compressed_size, desc.extra, seed_ptr, seed_len);
            chunk_out = Tokenizer::inverse_transform(tokenized.data(), tokenized.size());
        } else {
            throw std::runtime_error("Unsupported chunk compression method: " + std::to_string(desc.method));
        }

        if (chunk_out.size() != desc.uncompressed_size) {
            throw std::runtime_error("Chunk uncompressed size mismatch at chunk " + std::to_string(i));
        }

        // Verify Chunk CRC32
        uint32_t chunk_crc = CRC32::calculate(chunk_out.data(), chunk_out.size());
        if (chunk_crc != desc.checksum) {
            throw std::runtime_error("Chunk CRC32 checksum mismatch at chunk " + std::to_string(i));
        }

        // Update running SHA-256
        running_sha.update(chunk_out.data(), chunk_out.size());

        // Update seed tail for next chunk
        if (chunk_out.size() >= SEED_TAIL_SIZE) {
            seed_tail.assign(chunk_out.end() - SEED_TAIL_SIZE, chunk_out.end());
        } else {
            seed_tail = chunk_out;
        }

        // Stream chunk directly to disk
        out_file.write(reinterpret_cast<const char*>(chunk_out.data()), chunk_out.size());
        total_restored += chunk_out.size();
        curr_payload_offset += desc.compressed_size;

        if (verbose || (num_chunks > 10 && ((i + 1) % std::max<size_t>(1, num_chunks / 10) == 0 || (i + 1) == num_chunks))) {
            double pct = (100.0 * (i + 1)) / num_chunks;
            auto now = std::chrono::high_resolution_clock::now();
            double el = std::chrono::duration<double>(now - t0).count();
            double mb_s = el > 0 ? (total_restored / (1024.0 * 1024.0)) / el : 0.0;
            std::cout << "[PAX Progress] Decompressed chunks: " << (i + 1) << "/" << num_chunks
                      << " (" << std::fixed << std::setprecision(1) << pct << "%) - "
                      << mb_s << " MB/s\n" << std::flush;
        }
    }

    out_file.close();

    if (total_restored != expected_total_size) {
        throw std::runtime_error("Final uncompressed size mismatch");
    }

    // Verify Overall SHA-256 Checksum
    auto computed_sha = running_sha.finalize();
    if (std::memcmp(computed_sha.data(), header->sha256, 32) != 0) {
        throw std::runtime_error("SHA-256 integrity verification failed: file is corrupted or tampered");
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double mb = static_cast<double>(total_restored) / (1024.0 * 1024.0);
    double mb_s = elapsed_sec > 0.000001 ? (mb / elapsed_sec) : 0.0;

    std::cout << "[PAX] Restored: " << total_restored << " bytes in "
              << std::fixed << std::setprecision(2) << elapsed_sec << "s (" << mb_s << " MB/s)\n";
    std::cout << "[PAX] 100% Bit-Exact SHA-256 verified successfully: " << SHA256::to_hex(computed_sha) << "\n";

    return true;
}

VerificationResult PaxEngine::verify_file(const std::string& pax_path, bool verbose) {
    VerificationResult res;
    try {
        MappedFileView view(pax_path);
        size_t comp_len = view.size();
        const uint8_t* in = view.data();

        if (comp_len < sizeof(ContainerHeader) + sizeof(MAGIC_TAIL)) {
            res.error_message = "File too small for PAX container";
            return res;
        }

        if (std::memcmp(in, MAGIC_HEADER, 4) != 0) {
            res.error_message = "Magic header mismatch";
            return res;
        }

        if (std::memcmp(in + comp_len - 4, MAGIC_TAIL, 4) != 0) {
            res.error_message = "Tail sentinel mismatch (truncated file)";
            return res;
        }

        const ContainerHeader* header = reinterpret_cast<const ContainerHeader*>(in);
        res.uncompressed_size = header->uncompressed_size;
        res.compressed_size = comp_len;
        res.num_chunks = header->num_chunks;
        res.expected_sha256 = SHA256::to_hex(header->sha256);

        std::cout << "[PAX] Verifying container: " << pax_path << "\n";
        std::cout << "[PAX] Expected SHA-256: " << res.expected_sha256 << "\n";
        std::cout << "[PAX] Number of chunks: " << res.num_chunks << "\n";

        size_t table_offset = sizeof(ContainerHeader);
        size_t payload_offset = table_offset + res.num_chunks * sizeof(ChunkDescriptor);

        std::vector<ChunkDescriptor> chunk_descs(res.num_chunks);
        for (size_t i = 0; i < res.num_chunks; ++i) {
            std::memcpy(&chunk_descs[i], in + table_offset + i * sizeof(ChunkDescriptor), sizeof(ChunkDescriptor));
        }

        SHA256 running_sha;
        size_t curr_payload_offset = payload_offset;
        std::vector<uint8_t> seed_tail;

        for (size_t i = 0; i < res.num_chunks; ++i) {
            const auto& desc = chunk_descs[i];
            const uint8_t* payload_ptr = in + curr_payload_offset;
            std::vector<uint8_t> chunk_out;

            if (desc.method == METHOD_STORED) {
                chunk_out.assign(payload_ptr, payload_ptr + desc.compressed_size);
            } else if (desc.method == METHOD_BWT_RANS) {
                auto rle_data = RANS4::decompress(payload_ptr, desc.compressed_size);
                auto mtf_data = MTF::zero_rle_decode(rle_data.data(), rle_data.size());
                auto bwt_data = MTF::decode(mtf_data.data(), mtf_data.size());
                chunk_out = BWT::decode(bwt_data.data(), bwt_data.size(), desc.extra);
            } else if (desc.method == METHOD_TOKEN_BWT_RANS) {
                auto rle_data = RANS4::decompress(payload_ptr, desc.compressed_size);
                auto mtf_data = MTF::zero_rle_decode(rle_data.data(), rle_data.size());
                auto bwt_data = MTF::decode(mtf_data.data(), mtf_data.size());
                auto tokenized = BWT::decode(bwt_data.data(), bwt_data.size(), desc.extra);
                chunk_out = Tokenizer::inverse_transform(tokenized.data(), tokenized.size());
            } else if (desc.method == METHOD_ULTRA_MIXER) {
                const uint8_t* seed_ptr = seed_tail.empty() ? nullptr : seed_tail.data();
                size_t seed_len = seed_tail.size();
                chunk_out = ContextMixer::decompress(payload_ptr, desc.compressed_size, desc.uncompressed_size, seed_ptr, seed_len);
            } else if (desc.method == METHOD_TOKEN_ULTRA) {
                const uint8_t* seed_ptr = seed_tail.empty() ? nullptr : seed_tail.data();
                size_t seed_len = seed_tail.size();
                auto tokenized = ContextMixer::decompress(payload_ptr, desc.compressed_size, desc.extra, seed_ptr, seed_len);
                chunk_out = Tokenizer::inverse_transform(tokenized.data(), tokenized.size());
            }

            if (chunk_out.size() != desc.uncompressed_size) {
                throw std::runtime_error("Chunk uncompressed size mismatch at chunk " + std::to_string(i));
            }

            uint32_t chunk_crc = CRC32::calculate(chunk_out.data(), chunk_out.size());
            if (chunk_crc != desc.checksum) {
                throw std::runtime_error("Chunk CRC32 checksum mismatch at chunk " + std::to_string(i));
            }

            running_sha.update(chunk_out.data(), chunk_out.size());
            if (chunk_out.size() >= SEED_TAIL_SIZE) {
                seed_tail.assign(chunk_out.end() - SEED_TAIL_SIZE, chunk_out.end());
            } else {
                seed_tail = chunk_out;
            }

            curr_payload_offset += desc.compressed_size;

            if (verbose || (res.num_chunks > 10 && ((i + 1) % std::max<size_t>(1, res.num_chunks / 10) == 0 || (i + 1) == res.num_chunks))) {
                double pct = (100.0 * (i + 1)) / res.num_chunks;
                std::cout << "[PAX Progress] Verified chunks: " << (i + 1) << "/" << res.num_chunks
                          << " (" << std::fixed << std::setprecision(1) << pct << "%)\n" << std::flush;
            }
        }

        auto computed_sha = running_sha.finalize();
        res.computed_sha256 = SHA256::to_hex(computed_sha);

        if (res.computed_sha256 == res.expected_sha256) {
            res.success = true;
        } else {
            res.error_message = "SHA-256 checksum mismatch";
        }
    } catch (const std::exception& e) {
        res.error_message = e.what();
    }

    return res;
}

} // namespace pax
