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

inline bool is_alnum(uint8_t c) {
    return is_alpha(c) || (c >= '0' && c <= '9');
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

    // 2. Pass 1: Extract candidate words, Bi-words, MediaWiki syntax, XML tags
    std::unordered_map<std::string_view, uint32_t> counts;
    counts.reserve(524288);

    size_t i = 0;
    while (i < n) {
        uint8_t c = in[i];

        // MediaWiki ''' or ''
        if (c == '\'' && i + 1 < n && in[i + 1] == '\'') {
            if (i + 2 < n && in[i + 2] == '\'') {
                counts[std::string_view("'''", 3)]++;
                i += 3;
                continue;
            }
            counts[std::string_view("''", 2)]++;
            i += 2;
            continue;
        }

        // Web URLs: http:// or https://
        if (c == 'h' && i + 7 < n && std::memcmp(in + i, "http://", 7) == 0) {
            counts[std::string_view("http://", 7)]++;
            i += 7;
            continue;
        }
        if (c == 'h' && i + 8 < n && std::memcmp(in + i, "https://", 8) == 0) {
            counts[std::string_view("https://", 8)]++;
            i += 8;
            continue;
        }

        // Wikitable row / separator at newline: \n|- or \n| or \n* or \n#
        if (c == '\n' && i + 1 < n) {
            if (in[i + 1] == '|' && i + 2 < n && in[i + 2] == '-') {
                counts[std::string_view("\n|-", 3)]++;
                i += 3;
                continue;
            }
            if (in[i + 1] == '*' || in[i + 1] == '#' || in[i + 1] == '|' || in[i + 1] == '!') {
                counts[std::string_view(reinterpret_cast<const char*>(in + i), 2)]++;
                i += 2;
                continue;
            }
        }

        // [[Category: or [[
        if (c == '[' && i + 1 < n && in[i + 1] == '[') {
            size_t max_end = std::min(n, i + 48);
            size_t colon = i + 2;
            while (colon < max_end && is_alpha(in[colon])) colon++;
            if (colon < max_end && in[colon] == ':') {
                counts[std::string_view(reinterpret_cast<const char*>(in + i), colon + 1 - i)]++;
                i = colon + 1;
                continue;
            }
            counts[std::string_view("[[", 2)]++;
            i += 2;
            continue;
        }

        // ]]
        if (c == ']' && i + 1 < n && in[i + 1] == ']') {
            counts[std::string_view("]]", 2)]++;
            i += 2;
            continue;
        }

        // {{...}} or {{
        if (c == '{' && i + 1 < n && in[i + 1] == '{') {
            size_t max_end = std::min(n, i + 32);
            size_t end = i + 2;
            while (end < max_end && is_alpha(in[end])) end++;
            if (end - i >= 6) {
                counts[std::string_view(reinterpret_cast<const char*>(in + i), end - i)]++;
                i = end;
                continue;
            }
            counts[std::string_view("{{", 2)]++;
            i += 2;
            continue;
        }

        // }}
        if (c == '}' && i + 1 < n && in[i + 1] == '}') {
            counts[std::string_view("}}", 2)]++;
            i += 2;
            continue;
        }

        // ==
        if (c == '=' && i + 1 < n && in[i + 1] == '=') {
            size_t end = i + 2;
            while (end < n && in[end] == '=') end++;
            counts[std::string_view(reinterpret_cast<const char*>(in + i), end - i)]++;
            i = end;
            continue;
        }

        // XML tag <...>
        if (c == '<') {
            size_t max_end = std::min(n, i + 48);
            size_t end = i + 1;
            while (end < max_end && in[end] != '>' && in[end] != '\n' && in[end] != '\r') end++;
            if (end < max_end && in[end] == '>') {
                end++;
                counts[std::string_view(reinterpret_cast<const char*>(in + i), end - i)]++;
                i = end;
                continue;
            }
        }

        // HTML entity &...;
        if (c == '&') {
            size_t max_end = std::min(n, i + 12);
            size_t end = i + 1;
            while (end < max_end && (is_alnum(in[end]) || in[end] == '#')) end++;
            if (end < max_end && in[end] == ';' && (end - i) >= 3) {
                end++;
                counts[std::string_view(reinterpret_cast<const char*>(in + i), end - i)]++;
                i = end;
                continue;
            }
        }

        // Space + Bi-word or Space + Word
        if (c == ' ') {
            size_t max_end = std::min(n, i + 48);
            size_t end = i + 1;
            while (end < max_end && is_alpha(in[end])) end++;
            if (end - i >= 3) {
                counts[std::string_view(reinterpret_cast<const char*>(in + i), end - i)]++;
                if (end < n && in[end] == ' ') {
                    size_t w2_end = end + 1;
                    while (w2_end < max_end && is_alpha(in[w2_end])) w2_end++;
                    if (w2_end - end >= 3) {
                        counts[std::string_view(reinterpret_cast<const char*>(in + i), w2_end - i)]++;
                    }
                }
                i = end;
                continue;
            }
        }

        // Word
        if (is_alpha(c)) {
            size_t max_end = std::min(n, i + 32);
            size_t end = i + 1;
            while (end < max_end && is_alpha(in[end])) end++;
            if (end - i >= 3) {
                counts[std::string_view(reinterpret_cast<const char*>(in + i), end - i)]++;
                i = end;
                continue;
            }
        }

        i++;
    }

    // 3. Multi-tier token allocation: Direct single bytes, 2-byte escapes, 4-byte escapes
    // Core MediaWiki syntax tokens to prioritize into direct 1-byte slots if present
    static const char* const MW_CORE[] = {
        "[[", "]]", "{{", "}}", "==", "&quot;", "[[Category:", "'''", "''", "http://", "\n|-"
    };

    // Allocate ESC byte first
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

    // Step 1: Direct candidates (l >= 2) ranked by net savings: (l - 1) * cnt - (l + 2)
    std::vector<WordCandidate> direct_cands;
    direct_cands.reserve(counts.size());

    // First ensure high-frequency MediaWiki core syntax candidates are evaluated
    std::unordered_map<std::string_view, bool> mw_added;
    for (const char* mw : MW_CORE) {
        std::string_view sv(mw);
        auto it = counts.find(sv);
        if (it != counts.end() && it->second >= 4) {
            int32_t l = static_cast<int32_t>(sv.size());
            int32_t sav = (l - 1) * static_cast<int32_t>(it->second) - (l + 2);
            if (sav > 20) {
                direct_cands.push_back({std::string(sv), it->second, sav});
                mw_added[sv] = true;
            }
        }
    }

    for (const auto& [sv, cnt] : counts) {
        if (mw_added[sv]) continue;
        if (cnt < 4) continue;
        int32_t l = static_cast<int32_t>(sv.size());
        if (l < 2) continue;
        int32_t sav = (l - 1) * static_cast<int32_t>(cnt) - (l + 2);
        if (sav > 20) {
            direct_cands.push_back({std::string(sv), cnt, sav});
        }
    }

    if (direct_cands.empty()) {
        return {};
    }

    // Prioritize candidates: length 2 cannot be escaped, so give priority bonus equal to opportunity loss
    std::sort(direct_cands.begin(), direct_cands.end(), [](const WordCandidate& a, const WordCandidate& b) {
        int32_t score_a = a.saving + (a.word.size() == 2 ? 10000 : 0);
        int32_t score_b = b.saving + (b.word.size() == 2 ? 10000 : 0);
        return score_a > score_b;
    });

    size_t num_direct = std::min(unused_bytes.size(), direct_cands.size());
    std::vector<std::string> direct_words;
    std::unordered_map<std::string_view, uint8_t> word_to_direct;
    std::unordered_map<std::string_view, bool> is_chosen_direct;
    direct_words.reserve(num_direct);
    word_to_direct.reserve(num_direct);

    for (size_t k = 0; k < num_direct; ++k) {
        direct_words.push_back(direct_cands[k].word);
        word_to_direct[direct_words.back()] = unused_bytes[k];
        is_chosen_direct[direct_words.back()] = true;
    }

    // Step 2: Escaped candidates for remaining slots
    // 2-byte escape (l >= 3)
    std::vector<WordCandidate> esc2_cands;
    for (const auto& [sv, cnt] : counts) {
        if (is_chosen_direct.count(sv)) continue;
        if (cnt < 4) continue;
        int32_t l = static_cast<int32_t>(sv.size());
        if (l < 3) continue; // 2-byte escape requires length >= 3
        int32_t sav = (l - 2) * static_cast<int32_t>(cnt) - (l + 1);
        if (sav > 20) {
            esc2_cands.push_back({std::string(sv), cnt, sav});
        }
    }
    std::sort(esc2_cands.begin(), esc2_cands.end(), [](const WordCandidate& a, const WordCandidate& b) {
        return a.saving > b.saving;
    });

    std::vector<std::string> esc_words;
    std::unordered_map<std::string_view, uint16_t> word_to_esc;
    esc_words.reserve(8192);
    word_to_esc.reserve(8192);

    size_t num_esc2 = std::min<size_t>(254, esc2_cands.size());
    for (size_t k = 0; k < num_esc2; ++k) {
        uint16_t id = static_cast<uint16_t>(esc_words.size());
        esc_words.push_back(esc2_cands[k].word);
        word_to_esc[esc_words.back()] = id;
    }

    // 4-byte escape (l >= 5) to prevent short token inflation
    std::vector<WordCandidate> esc4_cands;
    for (size_t k = num_esc2; k < esc2_cands.size(); ++k) {
        const auto& c = esc2_cands[k];
        int32_t l = static_cast<int32_t>(c.word.size());
        if (l < 5) continue; // 4-byte escape requires length >= 5
        int32_t sav = (l - 4) * static_cast<int32_t>(c.count) - (l + 1);
        if (sav > 20) {
            esc4_cands.push_back({c.word, c.count, sav});
        }
    }
    std::sort(esc4_cands.begin(), esc4_cands.end(), [](const WordCandidate& a, const WordCandidate& b) {
        return a.saving > b.saving;
    });

    size_t num_esc4 = std::min<size_t>(8192 - esc_words.size(), esc4_cands.size());
    for (size_t k = 0; k < num_esc4; ++k) {
        uint16_t id = static_cast<uint16_t>(esc_words.size());
        esc_words.push_back(esc4_cands[k].word);
        word_to_esc[esc_words.back()] = id;
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

    // 6. Pass 2: Stream transformation with Longest Match First
    auto try_match = [&](std::string_view sv, size_t match_len) -> bool {
        auto it_dir = word_to_direct.find(sv);
        if (it_dir != word_to_direct.end()) {
            out.push_back(it_dir->second);
            i += match_len;
            return true;
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
            i += match_len;
            return true;
        }
        return false;
    };

    i = 0;
    while (i < n) {
        uint8_t c = in[i];

        // MediaWiki ''' or ''
        if (c == '\'' && i + 1 < n && in[i + 1] == '\'') {
            if (i + 2 < n && in[i + 2] == '\'' && try_match(std::string_view("'''", 3), 3)) continue;
            if (try_match(std::string_view("''", 2), 2)) continue;
        }

        // Web URLs: http:// or https://
        if (c == 'h' && i + 7 < n && std::memcmp(in + i, "http://", 7) == 0) {
            if (try_match(std::string_view("http://", 7), 7)) continue;
        }
        if (c == 'h' && i + 8 < n && std::memcmp(in + i, "https://", 8) == 0) {
            if (try_match(std::string_view("https://", 8), 8)) continue;
        }

        // Wikitable row / separator at newline: \n|- or \n| or \n* or \n#
        if (c == '\n' && i + 1 < n) {
            if (in[i + 1] == '|' && i + 2 < n && in[i + 2] == '-') {
                if (try_match(std::string_view("\n|-", 3), 3)) continue;
            }
            if (in[i + 1] == '*' || in[i + 1] == '#' || in[i + 1] == '|' || in[i + 1] == '!') {
                if (try_match(std::string_view(reinterpret_cast<const char*>(in + i), 2), 2)) continue;
            }
        }

        // [[Category: or [[
        if (c == '[' && i + 1 < n && in[i + 1] == '[') {
            size_t max_end = std::min(n, i + 48);
            size_t colon = i + 2;
            while (colon < max_end && is_alpha(in[colon])) colon++;
            if (colon < max_end && in[colon] == ':') {
                std::string_view sv(reinterpret_cast<const char*>(in + i), colon + 1 - i);
                if (try_match(sv, colon + 1 - i)) continue;
            }
            if (try_match(std::string_view("[[", 2), 2)) continue;
        }

        // ]]
        if (c == ']' && i + 1 < n && in[i + 1] == ']') {
            if (try_match(std::string_view("]]", 2), 2)) continue;
        }

        // {{...}} or {{
        if (c == '{' && i + 1 < n && in[i + 1] == '{') {
            size_t max_end = std::min(n, i + 32);
            size_t end = i + 2;
            while (end < max_end && is_alpha(in[end])) end++;
            if (end - i >= 6) {
                std::string_view sv(reinterpret_cast<const char*>(in + i), end - i);
                if (try_match(sv, end - i)) continue;
            }
            if (try_match(std::string_view("{{", 2), 2)) continue;
        }

        // }}
        if (c == '}' && i + 1 < n && in[i + 1] == '}') {
            if (try_match(std::string_view("}}", 2), 2)) continue;
        }

        // ==, ===, ==== (try longest first down to 2)
        if (c == '=' && i + 1 < n && in[i + 1] == '=') {
            size_t end = i + 2;
            while (end < n && in[end] == '=') end++;
            bool matched = false;
            for (size_t cur_len = end - i; cur_len >= 2; --cur_len) {
                if (try_match(std::string_view(reinterpret_cast<const char*>(in + i), cur_len), cur_len)) {
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        // XML tag <...>
        if (c == '<') {
            size_t max_end = std::min(n, i + 48);
            size_t end = i + 1;
            while (end < max_end && in[end] != '>' && in[end] != '\n' && in[end] != '\r') end++;
            if (end < max_end && in[end] == '>') {
                end++;
                std::string_view sv(reinterpret_cast<const char*>(in + i), end - i);
                if (try_match(sv, end - i)) continue;
            }
        }

        // HTML entity &...;
        if (c == '&') {
            size_t max_end = std::min(n, i + 12);
            size_t end = i + 1;
            while (end < max_end && (is_alnum(in[end]) || in[end] == '#')) end++;
            if (end < max_end && in[end] == ';' && (end - i) >= 3) {
                end++;
                std::string_view sv(reinterpret_cast<const char*>(in + i), end - i);
                if (try_match(sv, end - i)) continue;
            }
        }

        // Space + Bi-word or Space + Word (Longest first)
        if (c == ' ') {
            size_t max_end = std::min(n, i + 48);
            size_t end = i + 1;
            while (end < max_end && is_alpha(in[end])) end++;
            if (end - i >= 3) {
                if (end < n && in[end] == ' ') {
                    size_t w2_end = end + 1;
                    while (w2_end < max_end && is_alpha(in[w2_end])) w2_end++;
                    if (w2_end - end >= 3) {
                        std::string_view sv_bi(reinterpret_cast<const char*>(in + i), w2_end - i);
                        if (try_match(sv_bi, w2_end - i)) continue;
                    }
                }
                std::string_view sv(reinterpret_cast<const char*>(in + i), end - i);
                if (try_match(sv, end - i)) continue;
            }
        }

        // Word
        if (is_alpha(c)) {
            size_t max_end = std::min(n, i + 32);
            size_t end = i + 1;
            while (end < max_end && is_alpha(in[end])) end++;
            if (end - i >= 3) {
                std::string_view sv(reinterpret_cast<const char*>(in + i), end - i);
                if (try_match(sv, end - i)) continue;
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
