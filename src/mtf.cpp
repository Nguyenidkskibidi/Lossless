#include "pax/mtf.hpp"
#include <cstring>
#include <algorithm>

namespace pax {

std::vector<uint8_t> MTF::encode(const uint8_t* in, size_t n) {
    if (n == 0) return {};
    uint8_t list[256];
    for (int i = 0; i < 256; ++i) list[i] = static_cast<uint8_t>(i);

    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint8_t c = in[i];
        int idx = 0;
        while (list[idx] != c) ++idx;
        out[i] = static_cast<uint8_t>(idx);
        if (idx > 0) {
            std::memmove(list + 1, list, idx);
            list[0] = c;
        }
    }
    return out;
}

std::vector<uint8_t> MTF::decode(const uint8_t* in, size_t n) {
    if (n == 0) return {};
    uint8_t list[256];
    for (int i = 0; i < 256; ++i) list[i] = static_cast<uint8_t>(i);

    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint8_t idx = in[i];
        uint8_t c = list[idx];
        out[i] = c;
        if (idx > 0) {
            std::memmove(list + 1, list, idx);
            list[0] = c;
        }
    }
    return out;
}

std::vector<uint8_t> MTF::rle1_encode(const uint8_t* in, size_t n) {
    if (n == 0) return {};
    std::vector<uint8_t> out;
    out.reserve(n);

    size_t i = 0;
    while (i < n) {
        uint8_t c = in[i];
        size_t run = 1;
        while (i + run < n && in[i + run] == c && run < 259) {
            run++;
        }

        if (run < 4) {
            for (size_t r = 0; r < run; ++r) out.push_back(c);
            i += run;
        } else {
            out.push_back(c);
            out.push_back(c);
            out.push_back(c);
            out.push_back(c);
            out.push_back(static_cast<uint8_t>(run - 4));
            i += run;
        }
    }
    return out;
}

std::vector<uint8_t> MTF::rle1_decode(const uint8_t* in, size_t n) {
    if (n == 0) return {};
    std::vector<uint8_t> out;
    out.reserve(n);

    size_t i = 0;
    while (i < n) {
        uint8_t c = in[i++];
        out.push_back(c);

        if (out.size() >= 4 &&
            out[out.size() - 1] == c &&
            out[out.size() - 2] == c &&
            out[out.size() - 3] == c &&
            out[out.size() - 4] == c) {
            if (i < n) {
                uint8_t count = in[i++];
                for (size_t k = 0; k < count; ++k) {
                    out.push_back(c);
                }
            }
        }
    }
    return out;
}

std::vector<uint8_t> MTF::zero_rle_encode(const uint8_t* in, size_t n) {
    if (n == 0) return {};
    std::vector<uint8_t> out;
    out.reserve(n);

    size_t i = 0;
    while (i < n) {
        if (in[i] == 0) {
            size_t run = 0;
            while (i < n && in[i] == 0 && run < 257) {
                run++;
                i++;
            }
            if (run == 1) {
                out.push_back(0);
            } else {
                out.push_back(0);
                out.push_back(0);
                out.push_back(static_cast<uint8_t>(run - 2));
            }
        } else {
            out.push_back(in[i++]);
        }
    }
    return out;
}

std::vector<uint8_t> MTF::zero_rle_decode(const uint8_t* in, size_t n) {
    if (n == 0) return {};
    std::vector<uint8_t> out;
    out.reserve(n * 2);

    size_t i = 0;
    while (i < n) {
        if (in[i] == 0) {
            i++;
            if (i < n && in[i] == 0) {
                i++;
                uint8_t count = (i < n) ? in[i++] : 0;
                size_t total = 2 + count;
                out.insert(out.end(), total, 0);
            } else {
                out.push_back(0);
            }
        } else {
            out.push_back(in[i++]);
        }
    }
    return out;
}

} // namespace pax
