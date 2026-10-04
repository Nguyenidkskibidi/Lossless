#include "pax/common.hpp"
#include "pax/sha256.hpp"
#include "pax/crc32.hpp"
#include "pax/bwt.hpp"
#include "pax/mtf.hpp"
#include "pax/rans.hpp"
#include "pax/neural_mixer.hpp"
#include "pax/bit_arithmetic.hpp"
#include "pax/cabac.hpp"
#include "pax/tokenizer.hpp"
#include "pax/engine.hpp"
#include <iostream>
#include <cassert>
#include <random>
#include <string>
#include <vector>

#define PAX_ASSERT(expr) do { if (!(expr)) { std::cerr << "\nAssertion failed: " #expr "\n"; std::abort(); } } while(0)

void test_sha256() {
    std::cout << "[TEST] SHA-256 vectors... " << std::flush;
    auto h1 = pax::SHA256::hash("", 0);
    PAX_ASSERT(pax::SHA256::to_hex(h1) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    const char* s2 = "abc";
    auto h2 = pax::SHA256::hash(s2, 3);
    PAX_ASSERT(pax::SHA256::to_hex(h2) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    std::cout << "OK\n";
}

void test_crc32() {
    std::cout << "[TEST] CRC32 vectors... " << std::flush;
    const char* str = "The quick brown fox jumps over the lazy dog";
    uint32_t c = pax::CRC32::calculate(str, std::strlen(str));
    PAX_ASSERT(c == 0x414fa339);
    std::cout << "OK\n";
}

void test_bwt_roundtrip() {
    std::cout << "[TEST] BWT forward & inverse... " << std::flush;
    std::vector<std::string> cases = {
        "a", "banana", "abracadabra", "mmiissiissiippii",
        "Parallel Arithmetic Extreme Archiver BWT test string."
    };
    for (const auto& s : cases) {
        auto [L, p] = pax::BWT::encode((const uint8_t*)s.data(), s.size());
        auto rec = pax::BWT::decode(L.data(), L.size(), p);
        assert(rec.size() == s.size());
        assert(std::memcmp(rec.data(), s.data(), s.size()) == 0);
    }
    std::cout << "OK\n";
}

void test_mtf_roundtrip() {
    std::cout << "[TEST] MTF & Zero-RLE... " << std::flush;
    std::string text = "abracadabra banana banana mmiissiissiippii";
    auto enc = pax::MTF::encode((const uint8_t*)text.data(), text.size());
    auto dec = pax::MTF::decode(enc.data(), enc.size());
    assert(dec.size() == text.size());
    assert(std::memcmp(dec.data(), text.data(), text.size()) == 0);

    // Zero-run RLE
    std::vector<uint8_t> zeros = {0, 0, 0, 0, 1, 2, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 5};
    auto z_enc = pax::MTF::zero_rle_encode(zeros.data(), zeros.size());
    auto z_dec = pax::MTF::zero_rle_decode(z_enc.data(), z_enc.size());
    assert(z_dec == zeros);

    std::cout << "OK\n";
}

void test_rans_roundtrip() {
    std::cout << "[TEST] 4-way Interleaved rANS... " << std::flush;
    std::string text = "Project PAX 4-way interleaved SIMD rANS test string! 1234567890.";
    auto comp = pax::RANS4::compress((const uint8_t*)text.data(), text.size());
    auto dec = pax::RANS4::decompress(comp.data(), comp.size());
    assert(dec.size() == text.size());
    assert(std::memcmp(dec.data(), text.data(), text.size()) == 0);
    std::cout << "OK\n";
}

void test_cabac_roundtrip() {
    std::cout << "[TEST] CABAC MTF Coder... " << std::flush;
    std::vector<uint8_t> mtf_sample;
    // Generate realistic MTF sequence: runs of zeros, small ranks, occasional larger ranks
    for (int i = 0; i < 1000; ++i) {
        mtf_sample.push_back(0);
        mtf_sample.push_back(0);
        mtf_sample.push_back(0);
        mtf_sample.push_back(1);
        mtf_sample.push_back(0);
        mtf_sample.push_back(2);
        mtf_sample.push_back(i % 10 == 0 ? static_cast<uint8_t>(i % 256) : 0);
    }
    auto comp = pax::CabacMtf::compress(mtf_sample.data(), mtf_sample.size());
    auto dec = pax::CabacMtf::decompress(comp.data(), comp.size());
    PAX_ASSERT(dec == mtf_sample);
    std::cout << "OK (Shrank " << mtf_sample.size() << " -> " << comp.size() << " bytes)\n";
}

void test_context_mixer_roundtrip() {
    std::cout << "[TEST] Neural Context Mixer + SSE... " << std::flush;
    std::string text = "PAX extreme neural context mixing test. Order-1 to Order-32 Markov models + sparse contexts.";
    std::string seed = "Warm context seed data 1234567890.";

    auto comp = pax::ContextMixer::compress((const uint8_t*)text.data(), text.size(), (const uint8_t*)seed.data(), seed.size());
    auto dec = pax::ContextMixer::decompress(comp.data(), comp.size(), text.size(), (const uint8_t*)seed.data(), seed.size());
    assert(dec.size() == text.size());
    assert(std::memcmp(dec.data(), text.data(), text.size()) == 0);
    std::cout << "OK\n";
}

void test_end_to_end_engine() {
    std::cout << "[TEST] End-to-end Engine (Multi-chunk, Standard & Ultra)... " << std::flush;
    std::string sample = "Project PAX (Parallel Arithmetic Extreme Archiver) End-To-End Comprehensive Engine Verification!\n";
    std::vector<uint8_t> data;
    for (int i = 0; i < 500; ++i) {
        data.insert(data.end(), sample.begin(), sample.end());
    }

    // 1. Standard mode with multiple small chunks
    {
        pax::CompressionOptions opt;
        opt.ultra = false;
        opt.chunk_size = 16384; // 16 KB chunks to force multiple chunks
        auto comp = pax::PaxEngine::compress_memory(data.data(), data.size(), opt);
        auto rest = pax::PaxEngine::decompress_memory(comp.data(), comp.size());
        assert(rest == data);
    }

    // 2. Ultra mode with multiple small chunks and seed warming
    {
        pax::CompressionOptions opt;
        opt.ultra = true;
        opt.chunk_size = 16384;
        auto comp = pax::PaxEngine::compress_memory(data.data(), data.size(), opt);
        auto rest = pax::PaxEngine::decompress_memory(comp.data(), comp.size());
        assert(rest == data);
    }

    std::cout << "OK\n";
}

void test_tokenizer() {
    std::cout << "[TEST] Dynamic Tokenizer Pre-transform... " << std::flush;
    std::string sample =
        "<page>\n"
        "  <title>Anarchism</title>\n"
        "  <id>12</id>\n"
        "  <revision>\n"
        "    <id>12345</id>\n"
        "    <timestamp>2006-03-01T12:00:00Z</timestamp>\n"
        "    <contributor><username>SampleUser</username><id>999</id></contributor>\n"
        "    <text xml:space=\"preserve\">\n"
        "      Anarchism is a political philosophy that advocates self-governed societies based on voluntary institutions.\n"
        "      These are often described as stateless societies, although several authors have defined them more specifically.\n"
        "      Anarchism holds the state to be undesirable, unnecessary, and harmful.\n"
        "      Anarchism does not offer a fixed body of doctrine from a single world view, instead fluxing and flowing as a philosophy.\n"
        "      &quot;Direct action&quot; is central to many anarchist movements.\n"
        "      See &lt;http://en.wikipedia.org/wiki/Anarchism&gt; for more information &amp; references.\n"
        "    </text>\n"
        "  </revision>\n"
        "</page>\n";

    std::string text;
    for (int i = 0; i < 20; ++i) {
        text += sample;
    }

    auto transformed = pax::Tokenizer::transform(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    assert(!transformed.empty());
    assert(transformed.size() < text.size());

    auto restored = pax::Tokenizer::inverse_transform(transformed.data(), transformed.size());
    assert(restored.size() == text.size());
    assert(std::memcmp(restored.data(), text.data(), text.size()) == 0);

    std::cout << "OK (Shrank " << text.size() << " -> " << transformed.size() << " bytes, "
              << (100.0 * transformed.size()) / text.size() << "%)\n";
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  PROJECT PAX: COMPREHENSIVE ALGORITHMIC TEST SUITE\n";
    std::cout << "=================================================================\n";

    test_sha256();
    test_crc32();
    test_tokenizer();
    test_bwt_roundtrip();
    test_mtf_roundtrip();
    test_rans_roundtrip();
    test_cabac_roundtrip();
    test_context_mixer_roundtrip();
    test_end_to_end_engine();

    std::cout << "=================================================================\n";
    std::cout << "  ALL TEST SUITE CHECKS COMPLETED SUCCESSFULLY!\n";
    std::cout << "=================================================================\n";
    return 0;
}
