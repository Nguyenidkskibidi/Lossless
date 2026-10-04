#pragma once

#include "pax/common.hpp"
#include "pax/state_table.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <functional>
#include <cstring>

namespace pax {

constexpr int LOGIT_SCALE = 512;
constexpr int PROB_SCALE  = 4096; // 12-bit fixed point probability
constexpr int SUM_LIMIT   = 16384;
constexpr int NUM_MODELS  = 14;

struct FixedLookupTables {
    int16_t logit_lut[PROB_SCALE];
    uint16_t sigmoid_lut[2 * SUM_LIMIT + 1];

    FixedLookupTables();

    inline int16_t logit(uint16_t p) const {
        return logit_lut[std::clamp(static_cast<int>(p), 1, PROB_SCALE - 1)];
    }

    inline uint16_t sigmoid(int sum) const {
        int s = std::clamp(sum, -SUM_LIMIT, SUM_LIMIT);
        return sigmoid_lut[s + SUM_LIMIT];
    }
};

extern const FixedLookupTables g_fixed_tables;

template <size_t SIZE_BITS>
struct HashContextModel1B {
    static constexpr size_t TABLE_SIZE = 1ULL << SIZE_BITS;
    static constexpr size_t MASK = TABLE_SIZE - 1;
    std::vector<uint8_t> table;

    HashContextModel1B() : table(TABLE_SIZE, 0) {}

    void reset() {
        std::fill(table.begin(), table.end(), 0);
    }

    inline uint8_t get(uint64_t hash, uint8_t c0) const {
        size_t idx = (hash ^ (static_cast<uint64_t>(c0) * 2654435761ULL)) & MASK;
        return table[idx];
    }

    inline void update(uint64_t hash, uint8_t c0, int bit) {
        size_t idx = (hash ^ (static_cast<uint64_t>(c0) * 2654435761ULL)) & MASK;
        table[idx] = g_mahoney.next_state[table[idx]][bit];
    }
};

class SSETable1B {
public:
    static constexpr size_t BUCKETS = 32;

    SSETable1B();
    void reset();
    uint16_t estimate(uint16_t p_mix, uint8_t c0, size_t& out_bucket);
    void update(size_t bucket, uint8_t c0, int bit);

private:
    uint8_t table_[BUCKETS][256];
};

// Long Match Model with 64 MB history sliding window
class MatchModel {
public:
    static constexpr size_t HASH_BITS = 22; // 4M entries (~16 MB table)
    static constexpr size_t HASH_SIZE = 1ULL << HASH_BITS;
    static constexpr size_t HASH_MASK = HASH_SIZE - 1;
    static constexpr size_t MAX_HISTORY = 64 * 1024 * 1024; // 64 MB buffer

    MatchModel();
    void reset();

    void add_byte(uint8_t byte);
    inline size_t current_len() const { return match_len_; }
    uint16_t predict_bit(uint8_t c0) const;
    void update_bit(int bit);

private:
    std::vector<uint32_t> hash_table_;
    std::vector<uint8_t> history_;
    size_t hist_pos_ = 0;
    size_t match_pos_ = 0;
    size_t match_len_ = 0;
    uint32_t rolling_hash_ = 0;
    int bit_index_ = 7;
};

class ContextMixer {
public:
    using ProgressCallback = std::function<void(size_t done, size_t total)>;

    ContextMixer();
    void reset();

    void warm_seed(const uint8_t* seed_data, size_t seed_len);
    void add_byte_history(uint8_t byte);
    uint8_t get_prev(size_t offset) const;

    uint16_t predict_bit(uint8_t c0, uint16_t preds[NUM_MODELS], size_t& sse_bucket);
    void update(uint8_t c0, int bit, const uint16_t preds[NUM_MODELS], size_t sse_bucket, uint16_t p_final);

    static std::vector<uint8_t> compress(const uint8_t* in,
                                         size_t n,
                                         const uint8_t* seed = nullptr,
                                         size_t seed_len = 0,
                                         ProgressCallback progress = nullptr);

    static std::vector<uint8_t> decompress(const uint8_t* in,
                                           size_t comp_len,
                                           size_t orig_len,
                                           const uint8_t* seed = nullptr,
                                           size_t seed_len = 0,
                                           ProgressCallback progress = nullptr);

private:
    uint8_t order0_[256];
    uint8_t order1_[256][256];
    HashContextModel1B<24> model_order2_;  // 16M entries (16 MB)
    HashContextModel1B<24> model_order3_;  // 16M entries (16 MB)
    HashContextModel1B<24> model_order4_;  // 16M entries (16 MB)
    HashContextModel1B<24> model_order5_;  // 16M entries (16 MB)
    HashContextModel1B<24> model_order6_;  // 16M entries (16 MB)
    HashContextModel1B<24> model_order8_;  // 16M entries (16 MB)
    HashContextModel1B<22> model_sparse1_; // 4M entries (4 MB)
    HashContextModel1B<22> model_sparse2_; // 4M entries (4 MB)
    HashContextModel1B<22> model_sparse3_; // 4M entries (4 MB)
    HashContextModel1B<24> model_word_;    // 16M entries (16 MB)
    HashContextModel1B<24> model_xml_;     // 16M entries (16 MB)
    MatchModel match_model_;               // 64 MB history sliding window

    int32_t weights_[256][NUM_MODELS];
    SSETable1B sse_;

    uint8_t history_[64];
    size_t hist_pos_;

    uint64_t curr_word_hash_;
    uint64_t prev_word_hash_;

    uint32_t xml_tag_hash_;
    bool inside_tag_;
    bool inside_quotes_;
    int bracket_depth_;
};

} // namespace pax
