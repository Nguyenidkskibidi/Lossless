#include "pax/tokenizer.hpp"
#include <unordered_map>
#include <string_view>
#include <algorithm>
#include <cstring>
#include <array>

namespace pax {

bool Tokenizer::is_text(const uint8_t* in, size_t n) {
    if (n < 64) return false;
    size_t sample_len = std::min<size_t>(n, 8192);
    size_t text_chars = 0;
    size_t bad_controls = 0;

    for (size_t i = 0; i < sample_len; ++i) {
        uint8_t c = in[i];
        if (c >= 0x20 && c <= 0x7E) {
            text_chars++;
        } else if (c == '\t' || c == '\n' || c == '\r') {
            text_chars++;
        } else if (c >= 0x80) {
            text_chars++; // UTF-8 sequence
        } else if (c == 0x00 || c < 0x08 || (c > 0x0D && c < 0x1F)) {
            bad_controls++;
        }
    }

    if (bad_controls * 50 > sample_len) return false; // > 2% bad binary controls
    return (text_chars * 100) / sample_len >= 90;
}

namespace {

struct WordCandidate {
    std::string word;
    uint32_t count = 0;
    int32_t saving = 0;
};

inline bool is_alpha(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

} // anonymous namespace

std::vector<uint8_t> Tokenizer::transform(const uint8_t* in, size_t n) {
    if (!is_text(in, n) || n < 1024) {
        return {};
    }

    // 1. Analyze byte frequencies to find unused bytes for direct single-byte tokens and ESC byte
    std::array<uint32_t, 256> freqs{};
    for (size_t i = 0; i < n; ++i) {
        freqs[in[i]]++;
    }

    std::vector<uint8_t> unused_bytes;
    for (int b = 0; b < 256; ++b) {
        if (freqs[b] == 0) {
            unused_bytes.push_back(static_cast<uint8_t>(b));
        }
    }

    // 2. Pass 1: Extract candidate words and XML tags
    std::unordered_map<std::string_view, uint32_t> counts;
    counts.reserve(65536);

    size_t i = 0;
    while (i < n) {
        uint8_t c = in[i];
        if (c == '<') {
            size_t start = i;
            size_t max_end = std::min(n, start + 48);
            size_t end = start + 1;
            while (end < max_end && in[end] != '>' && in[end] != '\n' && in[end] != '\r') {
                end++;
            }
            if (end < max_end && in[end] == '>') {
                end++; // include '>'
                counts[std::string_view(reinterpret_cast<const char*>(in + start), end - start)]++;
                i = end;
                continue;
            }
        } else if (c == '&') {
            size_t start = i;
            size_t max_end = std::min(n, start + 12);
            size_t end = start + 1;
            while (end < max_end && is_alpha(in[end])) {
                end++;
            }
            if (end < max_end && in[end] == ';' && (end - start) >= 3) {
                end++; // include ';'
                counts[std::string_view(reinterpret_cast<const char*>(in + start), end - start)]++;
                i = end;
                continue;
            }
        } else if (c == ' ') {
            size_t start = i;
            size_t max_end = std::min(n, start + 32);
            size_t end = start + 1;
            while (end < max_end && is_alpha(in[end])) {
                end++;
            }
            if (end - start >= 3) {
                counts[std::string_view(reinterpret_cast<const char*>(in + start), end - start)]++;
                i = end;
                continue;
            }
        } else if (is_alpha(c)) {
            size_t start = i;
            size_t max_end = std::min(n, start + 32);
            size_t end = start + 1;
            while (end < max_end && is_alpha(in[end])) {
                end++;
            }
            if (end - start >= 3) {
                counts[std::string_view(reinterpret_cast<const char*>(in + start), end - start)]++;
                i = end;
                continue;
            }
        }
        i++;
    }

    // 3. Score and rank candidates by net compression savings
    std::vector<WordCandidate> candidates;
    candidates.reserve(counts.size());

    for (const auto& [sv, cnt] : counts) {
        if (cnt < 4) continue;
        int32_t l = static_cast<int32_t>(sv.size());
        int32_t sav = (l - 2) * static_cast<int32_t>(cnt) - (l + 2);
        if (sav > 50) {
            candidates.push_back({std::string(sv), cnt, sav});
        }
    }

    if (candidates.empty()) {
        return {};
    }

    std::sort(candidates.begin(), candidates.end(), [](const WordCandidate& a, const WordCandidate& b) {
        return a.saving > b.saving;
    });

    // 4. Allocate vocabulary: direct single bytes and escaped tokens
    uint8_t esc_byte = 0xFF;
    if (!unused_bytes.empty()) {
        esc_byte = unused_bytes.back();
        unused_bytes.pop_back();
    } else {
        uint32_t min_f = UINT32_MAX;
        for (int b = 0; b < 256; ++b) {
            if (freqs[b] < min_f) {
                min_f = freqs[b];
                esc_byte = static_cast<uint8_t>(b);
            }
        }
    }

    size_t num_direct = std::min(unused_bytes.size(), candidates.size());
    std::vector<std::string> direct_words;
    std::unordered_map<std::string_view, uint8_t> word_to_direct;
    direct_words.reserve(num_direct);
    word_to_direct.reserve(num_direct);

    for (size_t k = 0; k < num_direct; ++k) {
        direct_words.push_back(candidates[k].word);
        word_to_direct[direct_words.back()] = unused_bytes[k];
    }

    size_t max_esc = std::min<size_t>(candidates.size() - num_direct, 4096);
    std::vector<std::string> esc_words;
    std::unordered_map<std::string_view, uint16_t> word_to_esc;
    esc_words.reserve(max_esc);
    word_to_esc.reserve(max_esc);

    for (size_t k = 0; k < max_esc; ++k) {
        esc_words.push_back(candidates[num_direct + k].word);
        word_to_esc[esc_words.back()] = static_cast<uint16_t>(k);
    }

    // 5. Build output stream with compact vocabulary header
    std::vector<uint8_t> out;
    out.reserve(n);

    out.push_back(static_cast<uint8_t>(direct_words.size()));
    for (size_t k = 0; k < direct_words.size(); ++k) {
        out.push_back(unused_bytes[k]);
        out.push_back(static_cast<uint8_t>(direct_words[k].size()));
        const auto& w = direct_words[k];
        out.insert(out.end(), w.begin(), w.end());
    }

    out.push_back(esc_byte);
    uint16_t esc_cnt = static_cast<uint16_t>(esc_words.size());
    out.push_back(static_cast<uint8_t>(esc_cnt & 0xFF));
    out.push_back(static_cast<uint8_t>(esc_cnt >> 8));
    for (size_t k = 0; k < esc_words.size(); ++k) {
        out.push_back(static_cast<uint8_t>(esc_words[k].size()));
        const auto& w = esc_words[k];
        out.insert(out.end(), w.begin(), w.end());
    }

    // 6. Pass 2: Stream transformation
    i = 0;
    while (i < n) {
        uint8_t c = in[i];

        if (c == '<') {
            size_t start = i;
            size_t max_end = std::min(n, start + 48);
            size_t end = start + 1;
            while (end < max_end && in[end] != '>' && in[end] != '\n' && in[end] != '\r') {
                end++;
            }
            if (end < max_end && in[end] == '>') {
                end++;
                std::string_view sv(reinterpret_cast<const char*>(in + start), end - start);
                auto it_dir = word_to_direct.find(sv);
                if (it_dir != word_to_direct.end()) {
                    out.push_back(it_dir->second);
                    i = end;
                    continue;
                }
                auto it_esc = word_to_esc.find(sv);
                if (it_esc != word_to_esc.end()) {
                    uint16_t id = it_esc->second;
                    out.push_back(esc_byte);
                    if (id < 254) {
                        out.push_back(static_cast<uint8_t>(id + 1));
                    } else {
                        out.push_back(255);
                        out.push_back(static_cast<uint8_t>(id & 0xFF));
                        out.push_back(static_cast<uint8_t>(id >> 8));
                    }
                    i = end;
                    continue;
                }
            }
        } else if (c == '&') {
            size_t start = i;
            size_t max_end = std::min(n, start + 12);
            size_t end = start + 1;
            while (end < max_end && is_alpha(in[end])) {
                end++;
            }
            if (end < max_end && in[end] == ';' && (end - start) >= 3) {
                end++;
                std::string_view sv(reinterpret_cast<const char*>(in + start), end - start);
                auto it_dir = word_to_direct.find(sv);
                if (it_dir != word_to_direct.end()) {
                    out.push_back(it_dir->second);
                    i = end;
                    continue;
                }
                auto it_esc = word_to_esc.find(sv);
                if (it_esc != word_to_esc.end()) {
                    uint16_t id = it_esc->second;
                    out.push_back(esc_byte);
                    if (id < 254) {
                        out.push_back(static_cast<uint8_t>(id + 1));
                    } else {
                        out.push_back(255);
                        out.push_back(static_cast<uint8_t>(id & 0xFF));
                        out.push_back(static_cast<uint8_t>(id >> 8));
                    }
                    i = end;
                    continue;
                }
            }
        } else if (c == ' ') {
            size_t start = i;
            size_t max_end = std::min(n, start + 32);
            size_t end = start + 1;
            while (end < max_end && is_alpha(in[end])) {
                end++;
            }
            if (end - start >= 3) {
                std::string_view sv(reinterpret_cast<const char*>(in + start), end - start);
                auto it_dir = word_to_direct.find(sv);
                if (it_dir != word_to_direct.end()) {
                    out.push_back(it_dir->second);
                    i = end;
                    continue;
                }
                auto it_esc = word_to_esc.find(sv);
                if (it_esc != word_to_esc.end()) {
                    uint16_t id = it_esc->second;
                    out.push_back(esc_byte);
                    if (id < 254) {
                        out.push_back(static_cast<uint8_t>(id + 1));
                    } else {
                        out.push_back(255);
                        out.push_back(static_cast<uint8_t>(id & 0xFF));
                        out.push_back(static_cast<uint8_t>(id >> 8));
                    }
                    i = end;
                    continue;
                }
            }
        } else if (is_alpha(c)) {
            size_t start = i;
            size_t max_end = std::min(n, start + 32);
            size_t end = start + 1;
            while (end < max_end && is_alpha(in[end])) {
                end++;
            }
            if (end - start >= 3) {
                std::string_view sv(reinterpret_cast<const char*>(in + start), end - start);
                auto it_dir = word_to_direct.find(sv);
                if (it_dir != word_to_direct.end()) {
                    out.push_back(it_dir->second);
                    i = end;
                    continue;
                }
                auto it_esc = word_to_esc.find(sv);
                if (it_esc != word_to_esc.end()) {
                    uint16_t id = it_esc->second;
                    out.push_back(esc_byte);
                    if (id < 254) {
                        out.push_back(static_cast<uint8_t>(id + 1));
                    } else {
                        out.push_back(255);
                        out.push_back(static_cast<uint8_t>(id & 0xFF));
                        out.push_back(static_cast<uint8_t>(id >> 8));
                    }
                    i = end;
                    continue;
                }
            }
        }

        // Literal byte output
        if (c == esc_byte) {
            out.push_back(esc_byte);
            out.push_back(0); // 0 = literal esc
        } else {
            out.push_back(c);
        }
        i++;
    }

    if (out.size() >= n) {
        return {};
    }

    return out;
}

std::vector<uint8_t> Tokenizer::inverse_transform(const uint8_t* in, size_t n) {
    if (n < 4) return {};

    size_t pos = 0;

    // 1. Read direct table
    uint8_t direct_count = in[pos++];
    std::string direct_table[256];
    std::array<bool, 256> has_direct{};

    for (size_t k = 0; k < direct_count; ++k) {
        if (pos + 2 > n) return {};
        uint8_t byte_val = in[pos++];
        uint8_t len = in[pos++];
        if (pos + len > n) return {};
        direct_table[byte_val] = std::string(reinterpret_cast<const char*>(in + pos), len);
        has_direct[byte_val] = true;
        pos += len;
    }

    // 2. Read ESC table
    if (pos + 3 > n) return {};
    uint8_t esc_byte = in[pos++];
    uint16_t esc_cnt = static_cast<uint16_t>(in[pos]) | (static_cast<uint16_t>(in[pos + 1]) << 8);
    pos += 2;

    std::vector<std::string> esc_table(esc_cnt);
    for (size_t k = 0; k < esc_cnt; ++k) {
        if (pos + 1 > n) return {};
        uint8_t len = in[pos++];
        if (pos + len > n) return {};
        esc_table[k] = std::string(reinterpret_cast<const char*>(in + pos), len);
        pos += len;
    }

    // 3. Decode stream
    std::vector<uint8_t> out;
    out.reserve(n * 2);

    while (pos < n) {
        uint8_t c = in[pos++];
        if (c == esc_byte) {
            if (pos >= n) return {}; // Truncated
            uint8_t cmd = in[pos++];
            if (cmd == 0) {
                out.push_back(esc_byte);
            } else if (cmd <= 254) {
                uint16_t id = cmd - 1;
                if (id >= esc_table.size()) return {}; // Corrupt ID
                const auto& w = esc_table[id];
                out.insert(out.end(), w.begin(), w.end());
            } else {
                if (pos + 2 > n) return {}; // Truncated
                uint16_t id = static_cast<uint16_t>(in[pos]) | (static_cast<uint16_t>(in[pos + 1]) << 8);
                pos += 2;
                if (id >= esc_table.size()) return {}; // Corrupt ID
                const auto& w = esc_table[id];
                out.insert(out.end(), w.begin(), w.end());
            }
        } else if (has_direct[c]) {
            const auto& w = direct_table[c];
            out.insert(out.end(), w.begin(), w.end());
        } else {
            out.push_back(c);
        }
    }

    return out;
}

} // namespace pax
