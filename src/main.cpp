#include "pax/common.hpp"
#include "pax/engine.hpp"
#include "pax/sha256.hpp"
#include "pax/crc32.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

namespace {

void print_banner() {
    std::cout << "=================================================================\n";
    std::cout << "  PAX ENGINE (Parallel Arithmetic Extreme Archiver) v3.0.0\n";
    std::cout << "  High-Throughput Lossless Compression Architecture\n";
    std::cout << "  Features: Direct Byte Tokenizer + 64MB Long Match + Markov 2^24\n";
#if defined(PAX_HAS_NEON)
    std::cout << "  Hardware SIMD: ARM NEON + Hardware CRC32 + SHA256 (Enabled)\n";
#elif defined(PAX_HAS_AVX512)
    std::cout << "  Hardware SIMD: AVX-512 (Enabled)\n";
#elif defined(PAX_HAS_AVX2)
    std::cout << "  Hardware SIMD: AVX2 + FMA (Enabled)\n";
#else
    std::cout << "  Hardware SIMD: Generic C++20 Vectorized\n";
#endif
    std::cout << "=================================================================\n\n";
}

void print_usage(const char* prog) {
    print_banner();
    std::cout << "Usage:\n";
    std::cout << "  Compress:   " << prog << " -c <input_file> -o <output.pax> [-t <threads>] [-ultra] [-v]\n";
    std::cout << "  Decompress: " << prog << " -d <output.pax> -o <restored_file> [-v]\n";
    std::cout << "  Verify:     " << prog << " -t <output.pax> [-v]\n";
    std::cout << "  Self-Test:  " << prog << " --test\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c <file>     Input raw file to compress into .pax container\n";
    std::cout << "  -d <file>     Input .pax file to decompress\n";
    std::cout << "  -t <file>     Test & verify container integrity (SHA-256/CRC32) without disk write\n";
    std::cout << "  -o <file>     Output file destination path\n";
    std::cout << "  -t <num>      Number of worker threads (default: hardware concurrency)\n";
    std::cout << "  -b <mb>       Chunk size in megabytes (default: 4 MB, max: 64 MB)\n";
    std::cout << "  -ultra        Enable Extreme Neural Context Mixing (Multi-Order Markov + SSE)\n";
    std::cout << "  -v, --verbose Enable verbose progress and statistics\n";
    std::cout << "  --test        Run automated bit-exact safety verification test suite\n";
    std::cout << "  -h, --help    Show this help message\n";
}

int run_safety_tests();

} // anonymous namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode;
    std::string input_file;
    std::string output_file;
    pax::CompressionOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--test" || arg == "-test") {
            return run_safety_tests();
        } else if (arg == "-c" && i + 1 < argc) {
            mode = "compress";
            input_file = argv[++i];
        } else if (arg == "-d" && i + 1 < argc) {
            mode = "decompress";
            input_file = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            // Check if next arg is an integer (threads) or file path
            char* end = nullptr;
            long val = std::strtol(argv[i + 1], &end, 10);
            if (end != argv[i + 1] && *end == '\0') {
                options.threads = static_cast<int>(val);
                i++;
            } else {
                mode = "verify";
                input_file = argv[++i];
            }
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-b" && i + 1 < argc) {
            size_t mb = static_cast<size_t>(std::strtoul(argv[++i], nullptr, 10));
            if (mb > 0) {
                options.chunk_size = mb * 1024 * 1024;
            }
        } else if (arg == "-ultra" || arg == "--ultra") {
            options.ultra = true;
        } else if (arg == "-v" || arg == "--verbose") {
            options.verbose = true;
        } else {
            std::cerr << "[PAX Error] Unknown or unexpected argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        if (mode == "compress") {
            if (input_file.empty()) {
                std::cerr << "[PAX Error] No input file specified for compression (-c <file>)\n";
                return 1;
            }
            if (output_file.empty()) {
                output_file = input_file + ".pax";
            }
            bool ok = pax::PaxEngine::compress_file(input_file, output_file, options);
            return ok ? 0 : 1;
        } else if (mode == "decompress") {
            if (input_file.empty() || output_file.empty()) {
                std::cerr << "[PAX Error] Decompress requires both input (-d <file.pax>) and output (-o <dest>)\n";
                return 1;
            }
            bool ok = pax::PaxEngine::decompress_file(input_file, output_file, options.verbose);
            return ok ? 0 : 1;
        } else if (mode == "verify") {
            if (input_file.empty()) {
                std::cerr << "[PAX Error] No archive file specified for verification (-t <file.pax>)\n";
                return 1;
            }
            auto res = pax::PaxEngine::verify_file(input_file, true);
            if (res.success) {
                std::cout << "[PAX Integrity] SUCCESS: Archive is 100% bit-exact and valid.\n";
                std::cout << "  Payload SHA-256: " << res.computed_sha256 << "\n";
                std::cout << "  Uncompressed:   " << res.uncompressed_size << " bytes\n";
                std::cout << "  Compressed:     " << res.compressed_size << " bytes\n";
                std::cout << "  Total Chunks:   " << res.num_chunks << " (All passed CRC-32)\n";
                return 0;
            } else {
                std::cerr << "[PAX Integrity] FAILED: " << res.error_message << "\n";
                if (!res.expected_sha256.empty()) {
                    std::cerr << "  Expected SHA-256: " << res.expected_sha256 << "\n";
                    std::cerr << "  Computed SHA-256: " << res.computed_sha256 << "\n";
                }
                return 1;
            }
        } else {
            std::cerr << "[PAX Error] Please specify an action: -c (compress), -d (decompress), or -t (verify)\n";
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[PAX Exception] " << e.what() << "\n";
        return 1;
    }
}

namespace {

int run_safety_tests() {
    print_banner();
    std::cout << ">>> RUNNING AUTOMATED SAFETY VERIFICATION TEST SUITE <<<\n\n";

    struct TestCase {
        std::string name;
        std::vector<uint8_t> data;
    };

    std::vector<TestCase> test_cases;

    // 1. Empty file
    test_cases.push_back({"Empty File (0 bytes)", {}});

    // 2. Single byte
    test_cases.push_back({"Single Byte (1 byte)", {'X'}});

    // 3. Uniform repeated bytes
    test_cases.push_back({"Repeated Zeroes (64 KB)", std::vector<uint8_t>(65536, 0)});
    test_cases.push_back({"Repeated Character 'A' (128 KB)", std::vector<uint8_t>(128 * 1024, 'A')});

    // 4. Text data
    std::string text_sample =
        "PROJECT PAX: Parallel Arithmetic Extreme Archiver.\n"
        "Integrity Requirement: 100% Bit-Exact Lossless (Strict SHA-256 verification before and after round-trip).\n"
        "Multi-Order Markov Context Models: Order-1 up to Order-32 with hash-indexed state tables.\n"
        "Dynamic Weight Context Mixing (Logistic / Neural Mixing Layer) combined with Secondary Symbol Estimation (SSE).\n"
        "Burrows-Wheeler Transform (pBWT) with two-stage suffix sorting and 4-way interleaved rANS entropy coding.\n";
    std::vector<uint8_t> text_data;
    for (int i = 0; i < 200; ++i) {
        text_data.insert(text_data.end(), text_sample.begin(), text_sample.end());
    }
    test_cases.push_back({"Structured Text Corpus (Repetitive)", text_data});

    // 5. Binary pattern (counter / strided)
    std::vector<uint8_t> binary_data(100000);
    for (size_t i = 0; i < binary_data.size(); ++i) {
        binary_data[i] = static_cast<uint8_t>((i * 7 + (i >> 8)) & 0xFF);
    }
    test_cases.push_back({"Structured Binary Sequence", binary_data});

    bool all_passed = true;

    for (const auto& tc : test_cases) {
        std::cout << "[TEST] Running test: " << tc.name << " (" << tc.data.size() << " bytes)...\n";

        // Compute original hashes
        uint32_t orig_crc = pax::CRC32::calculate(tc.data.data(), tc.data.size());
        auto orig_sha = pax::SHA256::hash(tc.data.data(), tc.data.size());
        std::string orig_sha_hex = pax::SHA256::to_hex(orig_sha);

        // Test Standard Mode (BWT + MTF + rANS)
        {
            pax::CompressionOptions opt;
            opt.ultra = false;
            opt.chunk_size = 128 * 1024;

            auto comp = pax::PaxEngine::compress_memory(tc.data.data(), tc.data.size(), opt);
            auto restored = pax::PaxEngine::decompress_memory(comp.data(), comp.size());

            uint32_t rest_crc = pax::CRC32::calculate(restored.data(), restored.size());
            auto rest_sha = pax::SHA256::hash(restored.data(), restored.size());
            std::string rest_sha_hex = pax::SHA256::to_hex(rest_sha);

            bool crc_match = (orig_crc == rest_crc);
            bool sha_match = (orig_sha_hex == rest_sha_hex);
            bool exact_bytes = (tc.data == restored);

            if (crc_match && sha_match && exact_bytes) {
                std::cout << "  [Standard Mode] PASSED! Ratio: "
                          << (tc.data.empty() ? 0.0 : (100.0 * comp.size() / tc.data.size())) << "%\n";
            } else {
                std::cerr << "  [Standard Mode] FAILED!\n";
                all_passed = false;
            }
        }

        // Test Ultra Mode (Neural Context Mixer + SSE)
        {
            pax::CompressionOptions opt;
            opt.ultra = true;
            opt.chunk_size = 64 * 1024;

            auto comp = pax::PaxEngine::compress_memory(tc.data.data(), tc.data.size(), opt);
            auto restored = pax::PaxEngine::decompress_memory(comp.data(), comp.size());

            uint32_t rest_crc = pax::CRC32::calculate(restored.data(), restored.size());
            auto rest_sha = pax::SHA256::hash(restored.data(), restored.size());
            std::string rest_sha_hex = pax::SHA256::to_hex(rest_sha);

            bool crc_match = (orig_crc == rest_crc);
            bool sha_match = (orig_sha_hex == rest_sha_hex);
            bool exact_bytes = (tc.data == restored);

            if (crc_match && sha_match && exact_bytes) {
                std::cout << "  [Ultra Mode]    PASSED! Ratio: "
                          << (tc.data.empty() ? 0.0 : (100.0 * comp.size() / tc.data.size())) << "%\n";
            } else {
                std::cerr << "  [Ultra Mode]    FAILED!\n";
                all_passed = false;
            }
        }
    }

    std::cout << "\n=================================================================\n";
    if (all_passed) {
        std::cout << ">>> ALL AUTOMATED SAFETY VERIFICATION TESTS PASSED 100% <<<\n";
        std::cout << "CRC32(Original) == CRC32(Decompressed) [VERIFIED]\n";
        std::cout << "SHA256(Original) == SHA256(Decompressed) [VERIFIED]\n";
        std::cout << "=================================================================\n";
        return 0;
    } else {
        std::cerr << ">>> SOME TESTS FAILED! <<<\n";
        return 1;
    }
}

} // anonymous namespace
