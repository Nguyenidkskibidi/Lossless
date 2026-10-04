#pragma once

#include "pax/common.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

namespace pax {

class Tokenizer {
public:
    // Transforms input chunk into tokenized stream.
    // If chunk is not suitable for text tokenization (or produces no net gain), returns empty vector.
    static std::vector<uint8_t> transform(const uint8_t* in, size_t n);

    // Reverses tokenization back to bit-exact original data.
    static std::vector<uint8_t> inverse_transform(const uint8_t* in, size_t n);

    // Checks whether input chunk has characteristics of natural text / XML
    static bool is_text(const uint8_t* in, size_t n);
};

} // namespace pax
