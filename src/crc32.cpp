#include "pax/crc32.hpp"

namespace pax {

namespace {

// Slicing-by-8 CRC32 table initialization
struct SlicingTable {
    uint32_t table[8][256];
    constexpr SlicingTable() : table{} {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c >> 1) ^ (0xEDB88320 & -(c & 1));
            }
            table[0][i] = c;
        }
        for (int slice = 1; slice < 8; ++slice) {
            for (uint32_t i = 0; i < 256; ++i) {
                table[slice][i] = (table[slice - 1][i] >> 8) ^ table[0][table[slice - 1][i] & 0xFF];
            }
        }
    }
};

#if !defined(PAX_HAS_ARM_CRC32)
const SlicingTable g_crc_table;
#endif

} // anonymous namespace

CRC32::CRC32() : crc_(0xFFFFFFFF) {}

void CRC32::reset() {
    crc_ = 0xFFFFFFFF;
}

void CRC32::update(const void* data, size_t length) {
    const uint8_t* p = static_cast<const uint8_t*>(data);

#if defined(PAX_HAS_ARM_CRC32)
    // Hardware accelerated ARMv8-A CRC32 instructions
    while (length >= 8) {
        uint64_t val;
        std::memcpy(&val, p, sizeof(val));
        crc_ = __crc32d(crc_, val);
        p += 8;
        length -= 8;
    }
    while (length > 0) {
        crc_ = __crc32b(crc_, *p++);
        --length;
    }
#else
    // Fast slicing-by-8 software fallback
    while (length >= 8) {
        uint32_t one = *reinterpret_cast<const uint32_t*>(p) ^ crc_;
        uint32_t two = *reinterpret_cast<const uint32_t*>(p + 4);
        crc_ = g_crc_table.table[7][one & 0xFF] ^
               g_crc_table.table[6][(one >> 8) & 0xFF] ^
               g_crc_table.table[5][(one >> 16) & 0xFF] ^
               g_crc_table.table[4][(one >> 24) & 0xFF] ^
               g_crc_table.table[3][two & 0xFF] ^
               g_crc_table.table[2][(two >> 8) & 0xFF] ^
               g_crc_table.table[1][(two >> 16) & 0xFF] ^
               g_crc_table.table[0][(two >> 24) & 0xFF];
        p += 8;
        length -= 8;
    }
    while (length > 0) {
        crc_ = (crc_ >> 8) ^ g_crc_table.table[0][(crc_ ^ *p++) & 0xFF];
        --length;
    }
#endif
}

uint32_t CRC32::finalize() const {
    return crc_ ^ 0xFFFFFFFF;
}

uint32_t CRC32::calculate(const void* data, size_t length) {
    CRC32 hasher;
    hasher.update(data, length);
    return hasher.finalize();
}

} // namespace pax
