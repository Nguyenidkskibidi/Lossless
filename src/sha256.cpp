#include "pax/sha256.hpp"
#include <iomanip>
#include <sstream>

namespace pax {

namespace {

inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

[[maybe_unused]] inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

[[maybe_unused]] inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

[[maybe_unused]] inline uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

[[maybe_unused]] inline uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

[[maybe_unused]] inline uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

[[maybe_unused]] inline uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

constexpr uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

} // anonymous namespace

SHA256::SHA256() {
    reset();
}

void SHA256::reset() {
    state_[0] = 0x6a09e667;
    state_[1] = 0xbb67ae85;
    state_[2] = 0x3c6ef372;
    state_[3] = 0xa54ff53a;
    state_[4] = 0x510e527f;
    state_[5] = 0x9b05688c;
    state_[6] = 0x1f83d9ab;
    state_[7] = 0x5be0cd19;
    count_ = 0;
}

void SHA256::transform(const uint8_t block[BLOCK_SIZE]) {
#if defined(PAX_HAS_ARM_SHA2) && defined(__ARM_FEATURE_SHA2)
    uint32x4_t abcd = vld1q_u32(&state_[0]);
    uint32x4_t efgh = vld1q_u32(&state_[4]);

    uint32x4_t abcd_prev = abcd;
    uint32x4_t efgh_prev = efgh;

    // Load message schedule blocks and swap endianness to big-endian
    uint32x4_t msg0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 0)));
    uint32x4_t msg1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 16)));
    uint32x4_t msg2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 32)));
    uint32x4_t msg3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block + 48)));

    // Rounds 0-3
    uint32x4_t tmp = vaddq_u32(msg0, vld1q_u32(&K[0]));
    uint32x4_t abcd_next = vsha256hq_u32(abcd, efgh, tmp);
    efgh = vsha256h2q_u32(efgh, abcd, tmp);
    abcd = abcd_next;

    // Rounds 4-7
    tmp = vaddq_u32(msg1, vld1q_u32(&K[4]));
    abcd_next = vsha256hq_u32(abcd, efgh, tmp);
    efgh = vsha256h2q_u32(efgh, abcd, tmp);
    abcd = abcd_next;

    // Rounds 8-11
    tmp = vaddq_u32(msg2, vld1q_u32(&K[8]));
    abcd_next = vsha256hq_u32(abcd, efgh, tmp);
    efgh = vsha256h2q_u32(efgh, abcd, tmp);
    abcd = abcd_next;

    // Rounds 12-15
    tmp = vaddq_u32(msg3, vld1q_u32(&K[12]));
    abcd_next = vsha256hq_u32(abcd, efgh, tmp);
    efgh = vsha256h2q_u32(efgh, abcd, tmp);
    abcd = abcd_next;

    for (int i = 16; i < 64; i += 16) {
        // Schedule generation for 16 rounds
        msg0 = vsha256su0q_u32(msg0, msg1);
        msg0 = vsha256su1q_u32(msg0, msg2, msg3);
        tmp = vaddq_u32(msg0, vld1q_u32(&K[i]));
        abcd_next = vsha256hq_u32(abcd, efgh, tmp);
        efgh = vsha256h2q_u32(efgh, abcd, tmp);
        abcd = abcd_next;

        msg1 = vsha256su0q_u32(msg1, msg2);
        msg1 = vsha256su1q_u32(msg1, msg3, msg0);
        tmp = vaddq_u32(msg1, vld1q_u32(&K[i + 4]));
        abcd_next = vsha256hq_u32(abcd, efgh, tmp);
        efgh = vsha256h2q_u32(efgh, abcd, tmp);
        abcd = abcd_next;

        msg2 = vsha256su0q_u32(msg2, msg3);
        msg2 = vsha256su1q_u32(msg2, msg0, msg1);
        tmp = vaddq_u32(msg2, vld1q_u32(&K[i + 8]));
        abcd_next = vsha256hq_u32(abcd, efgh, tmp);
        efgh = vsha256h2q_u32(efgh, abcd, tmp);
        abcd = abcd_next;

        msg3 = vsha256su0q_u32(msg3, msg0);
        msg3 = vsha256su1q_u32(msg3, msg1, msg2);
        tmp = vaddq_u32(msg3, vld1q_u32(&K[i + 12]));
        abcd_next = vsha256hq_u32(abcd, efgh, tmp);
        efgh = vsha256h2q_u32(efgh, abcd, tmp);
        abcd = abcd_next;
    }

    abcd = vaddq_u32(abcd, abcd_prev);
    efgh = vaddq_u32(efgh, efgh_prev);

    vst1q_u32(&state_[0], abcd);
    vst1q_u32(&state_[4], efgh);

#else
    uint32_t W[64];
    for (int i = 0; i < 16; ++i) {
        W[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               (static_cast<uint32_t>(block[i * 4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
        W[i] = gamma1(W[i - 2]) + W[i - 7] + gamma0(W[i - 15]) + W[i - 16];
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
#endif
}

void SHA256::update(const void* data, size_t length) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    size_t buffer_idx = count_ & 63;
    count_ += length;

    if (buffer_idx > 0) {
        size_t needed = BLOCK_SIZE - buffer_idx;
        if (length < needed) {
            std::memcpy(buffer_ + buffer_idx, p, length);
            return;
        }
        std::memcpy(buffer_ + buffer_idx, p, needed);
        transform(buffer_);
        p += needed;
        length -= needed;
    }

    while (length >= BLOCK_SIZE) {
        transform(p);
        p += BLOCK_SIZE;
        length -= BLOCK_SIZE;
    }

    if (length > 0) {
        std::memcpy(buffer_, p, length);
    }
}

void SHA256::finalize(uint8_t digest[DIGEST_SIZE]) {
    uint64_t total_bits = count_ * 8;
    size_t buffer_idx = count_ & 63;

    buffer_[buffer_idx++] = 0x80;
    if (buffer_idx > 56) {
        std::memset(buffer_ + buffer_idx, 0, BLOCK_SIZE - buffer_idx);
        transform(buffer_);
        buffer_idx = 0;
    }
    std::memset(buffer_ + buffer_idx, 0, 56 - buffer_idx);

    // Append length in bits as 64-bit big-endian
    for (int i = 7; i >= 0; --i) {
        buffer_[56 + (7 - i)] = static_cast<uint8_t>((total_bits >> (i * 8)) & 0xFF);
    }
    transform(buffer_);

    for (int i = 0; i < 8; ++i) {
        digest[i * 4]     = static_cast<uint8_t>((state_[i] >> 24) & 0xFF);
        digest[i * 4 + 1] = static_cast<uint8_t>((state_[i] >> 16) & 0xFF);
        digest[i * 4 + 2] = static_cast<uint8_t>((state_[i] >> 8) & 0xFF);
        digest[i * 4 + 3] = static_cast<uint8_t>(state_[i] & 0xFF);
    }
}

std::array<uint8_t, SHA256::DIGEST_SIZE> SHA256::finalize() {
    std::array<uint8_t, DIGEST_SIZE> dig;
    finalize(dig.data());
    return dig;
}

std::array<uint8_t, SHA256::DIGEST_SIZE> SHA256::hash(const void* data, size_t length) {
    SHA256 hasher;
    hasher.update(data, length);
    return hasher.finalize();
}

std::string SHA256::to_hex(const uint8_t digest[DIGEST_SIZE]) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string out;
    out.reserve(DIGEST_SIZE * 2);
    for (size_t i = 0; i < DIGEST_SIZE; ++i) {
        out.push_back(hex_chars[(digest[i] >> 4) & 0x0F]);
        out.push_back(hex_chars[digest[i] & 0x0F]);
    }
    return out;
}

std::string SHA256::to_hex(const std::array<uint8_t, DIGEST_SIZE>& digest) {
    return to_hex(digest.data());
}

} // namespace pax
