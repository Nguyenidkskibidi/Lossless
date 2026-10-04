#include "pax/cabac.hpp"
#include "pax/bit_arithmetic.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace pax {

namespace {

// Adaptive Binary Context Model with fast exponential moving average
struct FastBitModel {
    uint16_t p1 = 2048; // Scaled to 4096 (0.5 initial probability)

    inline int get_prob() const { return p1; }

    inline void update(int bit) {
        // Shift 6 (1/64 adapt rate) achieves optimal compression on MTF streams
        if (bit == 1) {
            p1 += (4096 - p1) >> 6;
        } else {
            p1 -= p1 >> 6;
        }
        if (p1 < 4) p1 = 4;
        if (p1 > 4092) p1 = 4092;
    }
};

struct CatTable {
    uint8_t cat[256];
    uint8_t nbits[9];
    uint8_t base[9];

    CatTable() {
        nbits[0] = 0; base[0] = 0;
        nbits[1] = 0; base[1] = 1;
        nbits[2] = 1; base[2] = 2;
        nbits[3] = 2; base[3] = 4;
        nbits[4] = 3; base[4] = 8;
        nbits[5] = 4; base[5] = 16;
        nbits[6] = 5; base[6] = 32;
        nbits[7] = 6; base[7] = 64;
        nbits[8] = 7; base[8] = 128;

        for (int v = 0; v < 256; ++v) {
            if (v == 0) cat[v] = 0;
            else if (v == 1) cat[v] = 1;
            else if (v < 4) cat[v] = 2;
            else if (v < 8) cat[v] = 3;
            else if (v < 16) cat[v] = 4;
            else if (v < 32) cat[v] = 5;
            else if (v < 64) cat[v] = 6;
            else if (v < 128) cat[v] = 7;
            else cat[v] = 8;
        }
    }
};

static const CatTable g_cat;

inline int get_r_bin(uint8_t r) {
    if (r == 1) return 0;
    if (r == 2) return 1;
    if (r <= 4) return 2;
    if (r <= 8) return 3;
    if (r <= 16) return 4;
    return 5;
}

inline int get_z_bin(size_t z) {
    if (z == 1) return 0;
    if (z == 2) return 1;
    if (z <= 4) return 2;
    if (z <= 8) return 3;
    if (z <= 16) return 4;
    if (z <= 32) return 5;
    if (z <= 64) return 6;
    if (z <= 128) return 7;
    return 8;
}

} // anonymous namespace

std::vector<uint8_t> CabacMtf::compress(const uint8_t* in, size_t n) {
    if (n == 0) return {};

    std::vector<uint8_t> out;
    out.reserve(n / 2);

    // First 4 bytes: uncompressed MTF stream length (little-endian)
    out.push_back(static_cast<uint8_t>(n & 0xFF));
    out.push_back(static_cast<uint8_t>((n >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((n >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((n >> 24) & 0xFF));

    RangeBitEncoder enc;

    FastBitModel ctx_zero[16];
    FastBitModel ctx_one[8];
    FastBitModel ctx_cat[8][8];
    FastBitModel ctx_suffix[9][8];

    size_t z_streak = 0;
    uint8_t prev_r = 1;

    for (size_t i = 0; i < n; ++i) {
        uint8_t r = in[i];
        int r_bin = get_r_bin(prev_r);

        // 1. Context-Adaptive Zero-Decision
        int z_ctx = (z_streak == 0) ? r_bin : (6 + get_z_bin(z_streak));
        if (z_ctx > 15) z_ctx = 15;
        int is_zero = (r == 0) ? 1 : 0;
        enc.encode_bit(is_zero, ctx_zero[z_ctx].get_prob());
        ctx_zero[z_ctx].update(is_zero);

        if (is_zero) {
            z_streak++;
            continue;
        }

        z_streak = 0;

        // 2. Context-Adaptive One-Decision (r == 1)
        int is_one = (r == 1) ? 1 : 0;
        enc.encode_bit(is_one, ctx_one[r_bin].get_prob());
        ctx_one[r_bin].update(is_one);

        if (!is_one) {
            // r >= 2 -> value v = r - 2
            uint8_t v = r - 2;
            uint8_t c = g_cat.cat[v];

            // 3. Category Unary Decision Tree
            for (int k = 0; k < 8; ++k) {
                if (c == k) {
                    enc.encode_bit(0, ctx_cat[r_bin][k].get_prob());
                    ctx_cat[r_bin][k].update(0);
                    break;
                } else {
                    enc.encode_bit(1, ctx_cat[r_bin][k].get_prob());
                    ctx_cat[r_bin][k].update(1);
                }
            }

            // 4. Suffix Bits in Category
            uint8_t nbits = g_cat.nbits[c];
            if (nbits > 0) {
                uint8_t offset = v - g_cat.base[c];
                for (int b = nbits - 1; b >= 0; --b) {
                    int bit = (offset >> b) & 1;
                    enc.encode_bit(bit, ctx_suffix[c][b].get_prob());
                    ctx_suffix[c][b].update(bit);
                }
            }
        }

        prev_r = r;
    }

    enc.finish();
    const auto& bitstream = enc.get_buffer();
    out.insert(out.end(), bitstream.begin(), bitstream.end());
    return out;
}

std::vector<uint8_t> CabacMtf::decompress(const uint8_t* in, size_t comp_len) {
    if (comp_len < 4) return {};

    uint32_t original_n = read_le32(in);
    if (original_n == 0) return {};

    RangeBitDecoder dec(in + 4, comp_len - 4);
    std::vector<uint8_t> out;
    out.reserve(original_n);

    FastBitModel ctx_zero[16];
    FastBitModel ctx_one[8];
    FastBitModel ctx_cat[8][8];
    FastBitModel ctx_suffix[9][8];

    size_t z_streak = 0;
    uint8_t prev_r = 1;

    for (size_t i = 0; i < original_n; ++i) {
        int r_bin = get_r_bin(prev_r);
        int z_ctx = (z_streak == 0) ? r_bin : (6 + get_z_bin(z_streak));
        if (z_ctx > 15) z_ctx = 15;

        int is_zero = dec.decode_bit(ctx_zero[z_ctx].get_prob());
        ctx_zero[z_ctx].update(is_zero);

        if (is_zero) {
            out.push_back(0);
            z_streak++;
            continue;
        }

        z_streak = 0;
        int is_one = dec.decode_bit(ctx_one[r_bin].get_prob());
        ctx_one[r_bin].update(is_one);

        uint8_t r = 1;
        if (!is_one) {
            uint8_t c = 8;
            for (int k = 0; k < 8; ++k) {
                int bit = dec.decode_bit(ctx_cat[r_bin][k].get_prob());
                ctx_cat[r_bin][k].update(bit);
                if (bit == 0) {
                    c = k;
                    break;
                }
            }

            uint8_t nbits = g_cat.nbits[c];
            uint8_t offset = 0;
            if (nbits > 0) {
                for (int b = nbits - 1; b >= 0; --b) {
                    int bit = dec.decode_bit(ctx_suffix[c][b].get_prob());
                    ctx_suffix[c][b].update(bit);
                    offset |= (bit << b);
                }
            }
            r = g_cat.base[c] + offset + 2;
        }

        out.push_back(r);
        prev_r = r;
    }

    return out;
}

} // namespace pax
