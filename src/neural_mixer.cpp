#include "pax/neural_mixer.hpp"
#include "pax/bit_arithmetic.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace pax {

FixedLookupTables::FixedLookupTables() {
    for (int p = 0; p < PROB_SCALE; ++p) {
        double prob = std::clamp((p + 0.5) / PROB_SCALE, 0.0001, 0.9999);
        double val = std::log(prob / (1.0 - prob)) * LOGIT_SCALE;
        logit_lut[p] = static_cast<int16_t>(std::clamp(std::round(val), -2048.0, 2048.0));
    }
    for (int s = -SUM_LIMIT; s <= SUM_LIMIT; ++s) {
        double z = static_cast<double>(s) / LOGIT_SCALE;
        double prob = 1.0 / (1.0 + std::exp(-z));
        int p = static_cast<int>(std::round(prob * PROB_SCALE));
        sigmoid_lut[s + SUM_LIMIT] = static_cast<uint16_t>(std::clamp(p, 1, PROB_SCALE - 1));
    }
}

const FixedLookupTables g_fixed_tables;

SSETable::SSETable() {
    reset();
}

void SSETable::reset() {
    for (size_t b = 0; b < BUCKETS; ++b) {
        for (size_t c = 0; c < 256; ++c) {
            table_[b][c] = StateCounter{};
        }
    }
}

uint16_t SSETable::estimate(uint16_t p_mix, uint8_t c0, size_t& out_bucket) {
    size_t b = (static_cast<size_t>(p_mix) * BUCKETS) >> 12;
    if (b >= BUCKETS) b = BUCKETS - 1;
    out_bucket = b;

    const auto& sc = table_[b][c0];
    uint32_t total = sc.n0 + sc.n1;
    if (total < 4) {
        return p_mix; // Not enough samples, preserve sharp mixer probability!
    }

    uint16_t p_sse = sc.predict();
    uint32_t w = std::min<uint32_t>(16, total);
    uint32_t blended = (static_cast<uint32_t>(p_mix) * (32 - w) + static_cast<uint32_t>(p_sse) * w) >> 5;
    return static_cast<uint16_t>(std::clamp(blended, 1U, static_cast<uint32_t>(PROB_SCALE - 1)));
}

void SSETable::update(size_t bucket, uint8_t c0, int bit) {
    table_[bucket][c0].update(bit);
}

MatchModel::MatchModel() : hash_table_(HASH_SIZE, 0), history_(MAX_HISTORY, 0) {
    reset();
}

void MatchModel::reset() {
    std::fill(hash_table_.begin(), hash_table_.end(), 0);
    std::fill(history_.begin(), history_.end(), 0);
    hist_pos_ = 0;
    match_pos_ = 0;
    match_len_ = 0;
    rolling_hash_ = 0;
    bit_index_ = 7;
}

void MatchModel::add_byte(uint8_t byte) {
    history_[hist_pos_ % MAX_HISTORY] = byte;
    hist_pos_++;

    rolling_hash_ = ((rolling_hash_ << 5) + rolling_hash_) ^ byte;
    size_t h_idx = rolling_hash_ & HASH_MASK;

    if (match_len_ == 0) {
        uint32_t prev_pos = hash_table_[h_idx];
        if (prev_pos > 0 && prev_pos < hist_pos_) {
            size_t p1 = (hist_pos_ - 1) % MAX_HISTORY;
            size_t p2 = (prev_pos - 1) % MAX_HISTORY;
            if (history_[p1] == history_[p2]) {
                match_pos_ = prev_pos;
                match_len_ = 4;
            }
        }
    }
    hash_table_[h_idx] = static_cast<uint32_t>(hist_pos_);
    bit_index_ = 7;
}

uint16_t MatchModel::predict_bit(uint8_t c0) const {
    (void)c0;
    if (match_len_ < 4) {
        return 2048; // Neutral probability 0.5 (logit 0)
    }

    uint8_t expected_byte = history_[match_pos_ % MAX_HISTORY];
    int expected_bit = (expected_byte >> bit_index_) & 1;

    int conf = std::min(1950, static_cast<int>(match_len_ * 120));
    return expected_bit ? static_cast<uint16_t>(2048 + conf) : static_cast<uint16_t>(2048 - conf);
}

void MatchModel::update_bit(int bit) {
    if (match_len_ >= 4) {
        uint8_t expected_byte = history_[match_pos_ % MAX_HISTORY];
        int expected_bit = (expected_byte >> bit_index_) & 1;
        if (bit == expected_bit) {
            if (bit_index_ == 0) {
                match_pos_++;
                match_len_++;
            }
        } else {
            match_len_ = 0;
        }
    }
    bit_index_ = (bit_index_ - 1) & 7;
}

ContextMixer::ContextMixer() {
    reset();
}

void ContextMixer::reset() {
    for (int i = 0; i < 256; ++i) order0_[i] = StateCounter{};
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 256; ++j) order1_[i][j] = StateCounter{};
    }
    model_order2_.reset();
    model_order3_.reset();
    model_order4_.reset();
    model_order6_.reset();
    model_order8_.reset();
    model_order16_.reset();
    model_order32_.reset();
    model_sparse1_.reset();
    model_sparse2_.reset();
    model_sparse3_.reset();
    model_word_.reset();
    match_model_.reset();

    for (int i = 0; i < NUM_MODELS; ++i) weights_[i] = 1024;
    weights_[13] = 2048; // High weight for Long Match Model
    sse_.reset();
    std::memset(history_, 0, sizeof(history_));
    hist_pos_ = 0;
    hash_history_ = 0;
    word_hash_ = 0;
}

void ContextMixer::warm_seed(const uint8_t* seed_data, size_t seed_len) {
    if (!seed_data || seed_len == 0) return;
    size_t start = (seed_len > 64) ? (seed_len - 64) : 0;
    for (size_t i = start; i < seed_len; ++i) {
        add_byte_history(seed_data[i]);
    }
}

void ContextMixer::add_byte_history(uint8_t byte) {
    history_[hist_pos_ & 63] = byte;
    hist_pos_++;
    hash_history_ = (hash_history_ * 31337ULL) ^ (static_cast<uint64_t>(byte) + 1);

    if (byte == ' ' || byte == '\n' || byte == '<' || byte == '>') {
        word_hash_ = (word_hash_ * 1000003ULL) ^ static_cast<uint64_t>(byte);
    } else {
        word_hash_ = (word_hash_ * 33ULL) ^ static_cast<uint64_t>(byte);
    }

    match_model_.add_byte(byte);
}

uint8_t ContextMixer::get_prev(size_t offset) const {
    return history_[(hist_pos_ - offset) & 63];
}

uint16_t ContextMixer::predict_bit(uint8_t c0, uint16_t preds[NUM_MODELS], size_t& sse_bucket) {
    uint8_t b1 = get_prev(1);
    uint8_t b2 = get_prev(2);
    uint8_t b3 = get_prev(3);
    uint8_t b4 = get_prev(4);

    uint64_t h2 = (static_cast<uint64_t>(b2) << 8) | b1;
    uint64_t h3 = (h2 << 8) | b3;
    uint64_t h4 = (h3 << 8) | b4;
    uint64_t h6 = hash_history_ ^ (static_cast<uint64_t>(b1) * 1000003ULL);
    uint64_t h8 = hash_history_ * 33ULL;
    uint64_t h16 = (hash_history_ >> 16) ^ (hash_history_ << 48);
    uint64_t h32 = hash_history_ * 0x9E3779B97F4A7C15ULL;

    preds[0]  = order0_[c0].predict();
    preds[1]  = order1_[b1][c0].predict();
    preds[2]  = model_order2_.get(h2, c0).predict();
    preds[3]  = model_order3_.get(h3, c0).predict();
    preds[4]  = model_order4_.get(h4, c0).predict();
    preds[5]  = model_order6_.get(h6, c0).predict();
    preds[6]  = model_order8_.get(h8, c0).predict();
    preds[7]  = model_order16_.get(h16, c0).predict();
    preds[8]  = model_order32_.get(h32, c0).predict();
    preds[9]  = model_sparse1_.get(b2, c0).predict();
    preds[10] = model_sparse2_.get(b3, c0).predict();
    preds[11] = model_sparse3_.get(b4, c0).predict();
    preds[12] = model_word_.get(word_hash_, c0).predict();
    preds[13] = match_model_.predict_bit(c0);

    int32_t sum = 0;
    for (int k = 0; k < NUM_MODELS; ++k) {
        int16_t l = g_fixed_tables.logit(preds[k]);
        sum += (weights_[k] * static_cast<int32_t>(l)) >> 10;
    }

    uint16_t p_mix = g_fixed_tables.sigmoid(sum);
    uint16_t p_final = sse_.estimate(p_mix, c0, sse_bucket);
    return p_final;
}

void ContextMixer::update(uint8_t c0, int bit, const uint16_t preds[NUM_MODELS], size_t sse_bucket, uint16_t p_final) {
    sse_.update(sse_bucket, c0, bit);
    match_model_.update_bit(bit);

    uint8_t b1 = get_prev(1);
    uint8_t b2 = get_prev(2);
    uint8_t b3 = get_prev(3);
    uint8_t b4 = get_prev(4);

    uint64_t h2 = (static_cast<uint64_t>(b2) << 8) | b1;
    uint64_t h3 = (h2 << 8) | b3;
    uint64_t h4 = (h3 << 8) | b4;
    uint64_t h6 = hash_history_ ^ (static_cast<uint64_t>(b1) * 1000003ULL);
    uint64_t h8 = hash_history_ * 33ULL;
    uint64_t h16 = (hash_history_ >> 16) ^ (hash_history_ << 48);
    uint64_t h32 = hash_history_ * 0x9E3779B97F4A7C15ULL;

    order0_[c0].update(bit);
    order1_[b1][c0].update(bit);
    model_order2_.get(h2, c0).update(bit);
    model_order3_.get(h3, c0).update(bit);
    model_order4_.get(h4, c0).update(bit);
    model_order6_.get(h6, c0).update(bit);
    model_order8_.get(h8, c0).update(bit);
    model_order16_.get(h16, c0).update(bit);
    model_order32_.get(h32, c0).update(bit);
    model_sparse1_.get(b2, c0).update(bit);
    model_sparse2_.get(b3, c0).update(bit);
    model_sparse3_.get(b4, c0).update(bit);
    model_word_.get(word_hash_, c0).update(bit);

    int32_t error = (bit ? PROB_SCALE : 0) - static_cast<int32_t>(p_final);
    for (int k = 0; k < NUM_MODELS; ++k) {
        int16_t l = g_fixed_tables.logit(preds[k]);
        int32_t dw = (error * static_cast<int32_t>(l)) >> 14;
        weights_[k] = std::clamp(weights_[k] + dw, -32768, 32767);
    }
}

std::vector<uint8_t> ContextMixer::compress(const uint8_t* in, size_t n, const uint8_t* seed, size_t seed_len) {
    if (n == 0) return {};

    RangeBitEncoder enc;
    ContextMixer mixer;
    if (seed && seed_len > 0) {
        mixer.warm_seed(seed, seed_len);
    }

    uint16_t preds[NUM_MODELS];
    size_t sse_bucket = 0;

    for (size_t i = 0; i < n; ++i) {
        uint8_t byte = in[i];
        uint8_t c0 = 1;
        for (int b = 7; b >= 0; --b) {
            int bit = (byte >> b) & 1;
            uint16_t p1 = mixer.predict_bit(c0, preds, sse_bucket);
            enc.encode_bit(bit, p1);
            mixer.update(c0, bit, preds, sse_bucket, p1);
            c0 = (c0 << 1) | bit;
        }
        mixer.add_byte_history(byte);
    }
    enc.finish();
    return enc.release_buffer();
}

std::vector<uint8_t> ContextMixer::decompress(const uint8_t* in, size_t comp_len, size_t orig_len, const uint8_t* seed, size_t seed_len) {
    if (orig_len == 0) return {};

    RangeBitDecoder dec(in, comp_len);
    ContextMixer mixer;
    if (seed && seed_len > 0) {
        mixer.warm_seed(seed, seed_len);
    }

    uint16_t preds[NUM_MODELS];
    size_t sse_bucket = 0;

    std::vector<uint8_t> out;
    out.reserve(orig_len);

    for (size_t i = 0; i < orig_len; ++i) {
        uint8_t byte = 0;
        uint8_t c0 = 1;
        for (int b = 7; b >= 0; --b) {
            uint16_t p1 = mixer.predict_bit(c0, preds, sse_bucket);
            int bit = dec.decode_bit(p1);
            byte |= (bit << b);
            mixer.update(c0, bit, preds, sse_bucket, p1);
            c0 = (c0 << 1) | bit;
        }
        mixer.add_byte_history(byte);
        out.push_back(byte);
    }
    return out;
}

} // namespace pax
