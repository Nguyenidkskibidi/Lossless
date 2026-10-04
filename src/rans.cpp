#include "pax/rans.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace pax {

void RANS4::FrequencyTable::build(const uint32_t counts[256], size_t total_count) {
    if (total_count == 0) {
        std::memset(freq, 0, sizeof(freq));
        std::memset(cum, 0, sizeof(cum));
        return;
    }

    uint32_t sum = 0;
    for (int i = 0; i < 256; ++i) {
        if (counts[i] > 0) {
            uint32_t f = static_cast<uint32_t>((static_cast<uint64_t>(counts[i]) * (SCALE_M - 256)) / total_count) + 1;
            freq[i] = static_cast<uint16_t>(f);
            sum += f;
        } else {
            freq[i] = 0;
        }
    }

    int diff = static_cast<int>(SCALE_M) - static_cast<int>(sum);
    int max_idx = 0;
    for (int i = 1; i < 256; ++i) {
        if (freq[i] > freq[max_idx]) max_idx = i;
    }
    freq[max_idx] += diff;

    cum[0] = 0;
    for (int i = 0; i < 256; ++i) {
        cum[i + 1] = cum[i] + freq[i];
        for (uint16_t s = cum[i]; s < cum[i + 1]; ++s) {
            lut[s] = static_cast<uint8_t>(i);
        }
    }
}

void RANS4::FrequencyTable::serialize(std::vector<uint8_t>& out) const {
    uint8_t mask[32] = {};
    int active = 0;
    for (int i = 0; i < 256; ++i) {
        if (freq[i] > 0) {
            mask[i / 8] |= (1 << (i % 8));
            active++;
        }
    }
    out.insert(out.end(), mask, mask + 32);

    if (active > 1) {
        for (int i = 0; i < 256; ++i) {
            if (freq[i] > 0) {
                out.push_back(static_cast<uint8_t>(freq[i] & 0xFF));
                out.push_back(static_cast<uint8_t>((freq[i] >> 8) & 0xFF));
            }
        }
    }
}

bool RANS4::FrequencyTable::deserialize(const uint8_t*& in_ptr, const uint8_t* in_end) {
    if (in_ptr + 32 > in_end) return false;
    const uint8_t* mask = in_ptr;
    in_ptr += 32;

    int active = 0;
    int single_sym = -1;
    for (int i = 0; i < 256; ++i) {
        if (mask[i / 8] & (1 << (i % 8))) {
            active++;
            single_sym = i;
        }
    }

    std::memset(freq, 0, sizeof(freq));
    if (active == 0) {
        std::memset(cum, 0, sizeof(cum));
        return true;
    }

    if (active == 1) {
        freq[single_sym] = SCALE_M;
    } else {
        if (in_ptr + active * 2 > in_end) return false;
        for (int i = 0; i < 256; ++i) {
            if (mask[i / 8] & (1 << (i % 8))) {
                uint16_t f = static_cast<uint16_t>(in_ptr[0]) |
                            (static_cast<uint16_t>(in_ptr[1]) << 8);
                in_ptr += 2;
                freq[i] = f;
            }
        }
    }

    cum[0] = 0;
    for (int i = 0; i < 256; ++i) {
        cum[i + 1] = cum[i] + freq[i];
        for (uint16_t s = cum[i]; s < cum[i + 1]; ++s) {
            lut[s] = static_cast<uint8_t>(i);
        }
    }
    return (cum[256] == SCALE_M);
}

std::vector<uint8_t> RANS4::compress(const uint8_t* in, size_t n) {
    if (n == 0) return {};

    std::vector<uint8_t> out;
    out.reserve(n + 64);

    // Write original uncompressed length (4 bytes little-endian)
    uint8_t len_bytes[4];
    write_le32(len_bytes, static_cast<uint32_t>(n));
    out.insert(out.end(), len_bytes, len_bytes + 4);

    uint32_t counts[256] = {};
    for (size_t i = 0; i < n; ++i) counts[in[i]]++;

    FrequencyTable ft;
    ft.build(counts, n);
    ft.serialize(out);

    // If only 1 unique symbol exists
    int active = 0;
    for (int i = 0; i < 256; ++i) {
        if (ft.freq[i] > 0) active++;
    }
    if (active <= 1) return out;

    uint32_t s[4] = { RANS_L, RANS_L, RANS_L, RANS_L };
    std::vector<uint8_t> buf;
    buf.reserve(n + 64);

    // Encode in reverse order across 4 interleaved lanes
    for (size_t i = n; i > 0; --i) {
        size_t idx = i - 1;
        int lane = idx & 3;
        uint8_t sym = in[idx];
        uint32_t f = ft.freq[sym];
        uint32_t c = ft.cum[sym];

        uint32_t x_max = ((RANS_L >> SCALE_BITS) << 8) * f;
        while (s[lane] >= x_max) {
            buf.push_back(static_cast<uint8_t>(s[lane] & 0xFF));
            s[lane] >>= 8;
        }
        s[lane] = ((s[lane] / f) << SCALE_BITS) + (s[lane] % f) + c;
    }

    // Append 4 final 32-bit states (16 bytes)
    for (int l = 0; l < 4; ++l) {
        out.push_back(static_cast<uint8_t>(s[l] & 0xFF));
        out.push_back(static_cast<uint8_t>((s[l] >> 8) & 0xFF));
        out.push_back(static_cast<uint8_t>((s[l] >> 16) & 0xFF));
        out.push_back(static_cast<uint8_t>((s[l] >> 24) & 0xFF));
    }

    // Append reversed byte stream so decoder reads forward
    out.insert(out.end(), buf.rbegin(), buf.rend());
    return out;
}

std::vector<uint8_t> RANS4::decompress(const uint8_t* in, size_t comp_len) {
    if (comp_len < 4) return {};

    uint32_t orig_len = read_le32(in);
    if (orig_len == 0) return {};

    const uint8_t* in_ptr = in + 4;
    const uint8_t* in_end = in + comp_len;

    FrequencyTable ft;
    if (!ft.deserialize(in_ptr, in_end)) {
        throw std::runtime_error("Malformed rANS frequency table");
    }

    // Check if single symbol
    int single_sym = -1;
    int active = 0;
    for (int i = 0; i < 256; ++i) {
        if (ft.freq[i] > 0) {
            active++;
            single_sym = i;
        }
    }
    if (active == 1) {
        return std::vector<uint8_t>(orig_len, static_cast<uint8_t>(single_sym));
    }

    if (in_ptr + 16 > in_end) {
        throw std::runtime_error("Truncated rANS payload header");
    }

    uint32_t s[4];
    for (int l = 0; l < 4; ++l) {
        s[l] = static_cast<uint32_t>(in_ptr[l * 4]) |
              (static_cast<uint32_t>(in_ptr[l * 4 + 1]) << 8) |
              (static_cast<uint32_t>(in_ptr[l * 4 + 2]) << 16) |
              (static_cast<uint32_t>(in_ptr[l * 4 + 3]) << 24);
    }
    in_ptr += 16;

    std::vector<uint8_t> out(orig_len);

#if defined(PAX_HAS_NEON)
    // SIMD vectorization on ARM NEON
    size_t i = 0;
    size_t vec_end = orig_len >= 4 ? orig_len - 3 : 0;
    while (i < vec_end) {
        // 4 lanes unrolled
        uint32_t slot0 = s[0] & (SCALE_M - 1);
        uint32_t slot1 = s[1] & (SCALE_M - 1);
        uint32_t slot2 = s[2] & (SCALE_M - 1);
        uint32_t slot3 = s[3] & (SCALE_M - 1);

        uint8_t sym0 = ft.lut[slot0];
        uint8_t sym1 = ft.lut[slot1];
        uint8_t sym2 = ft.lut[slot2];
        uint8_t sym3 = ft.lut[slot3];

        out[i + 0] = sym0;
        out[i + 1] = sym1;
        out[i + 2] = sym2;
        out[i + 3] = sym3;

        s[0] = ft.freq[sym0] * (s[0] >> SCALE_BITS) + (slot0 - ft.cum[sym0]);
        s[1] = ft.freq[sym1] * (s[1] >> SCALE_BITS) + (slot1 - ft.cum[sym1]);
        s[2] = ft.freq[sym2] * (s[2] >> SCALE_BITS) + (slot2 - ft.cum[sym2]);
        s[3] = ft.freq[sym3] * (s[3] >> SCALE_BITS) + (slot3 - ft.cum[sym3]);

        while (s[0] < RANS_L && in_ptr < in_end) { s[0] = (s[0] << 8) | (*in_ptr++); }
        while (s[1] < RANS_L && in_ptr < in_end) { s[1] = (s[1] << 8) | (*in_ptr++); }
        while (s[2] < RANS_L && in_ptr < in_end) { s[2] = (s[2] << 8) | (*in_ptr++); }
        while (s[3] < RANS_L && in_ptr < in_end) { s[3] = (s[3] << 8) | (*in_ptr++); }
        i += 4;
    }
    while (i < orig_len) {
        int lane = i & 3;
        uint32_t slot = s[lane] & (SCALE_M - 1);
        uint8_t sym = ft.lut[slot];
        out[i] = sym;
        s[lane] = ft.freq[sym] * (s[lane] >> SCALE_BITS) + (slot - ft.cum[sym]);
        while (s[lane] < RANS_L && in_ptr < in_end) {
            s[lane] = (s[lane] << 8) | (*in_ptr++);
        }
        i++;
    }
#else
    for (size_t i = 0; i < orig_len; ++i) {
        int lane = i & 3;
        uint32_t slot = s[lane] & (SCALE_M - 1);
        uint8_t sym = ft.lut[slot];
        out[i] = sym;

        s[lane] = ft.freq[sym] * (s[lane] >> SCALE_BITS) + (slot - ft.cum[sym]);
        while (s[lane] < RANS_L && in_ptr < in_end) {
            s[lane] = (s[lane] << 8) | (*in_ptr++);
        }
    }
#endif

    return out;
}

} // namespace pax
