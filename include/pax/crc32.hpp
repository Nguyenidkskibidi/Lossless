#pragma once

#include "pax/common.hpp"

namespace pax {

class CRC32 {
public:
    CRC32();
    void reset();
    void update(const void* data, size_t length);
    uint32_t finalize() const;

    static uint32_t calculate(const void* data, size_t length);

private:
    uint32_t crc_;
};

} // namespace pax
