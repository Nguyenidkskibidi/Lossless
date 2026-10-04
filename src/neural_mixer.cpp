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

SSETable1B::SSETable1B() {
    reset();
}

void SSETable1B::reset() {
    std::memset(table_, 0, sizeof(table_));
}

uint16_t SSETable1B::estimate(uint16_t p_mix, uint8_t c0, size_t& out_bucket) {
    size_t b = (static_cast<size_t>(p_mix) * BUCKETS) >> 12;
    if (b >= BUCKETS) b = BUCKETS - 1;
    out_bucket = b;

    uint8_t state = table_[b][c0];
    if (state == 0) return p_mix;

    uint16_t p_sse = g_mahoney.state_prob[state];
    int n0 = MAHONEY_STATE_TABLE[state][2];
    int n1 = MAHONEY_STATE_TABLE[state][3];
    uint32_t total = n0 + n1;
    uint32_t w = std::min<uint32_t>(16, total * 2);
    uint32_t blended = (static_cast<uint32_t>(p_mix) * (32 - w) + static_cast<uint32_t>(p_sse) * w) >> 5;
    return static_cast<uint16_t>(std::clamp(blended, 1U, static_cast<uint32_t>(PROB_SCALE - 1)));
}

void SSETable1B::update(size_t bucket, uint8_t c0, int bit) {
    table_[bucket][c0] = g_mahoney.next_state[table_[bucket][c0]][bit];
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
        return 2048;
    }

    uint8_t expected_byte = history_[match_pos_ % MAX_HISTORY];
    int expected_bit = (expected_byte >> bit_index_) & 1;

    int conf = std::min(2040, static_cast<int>(match_len_ * 180));
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
    std::memset(order0_, 0, sizeof(order0_));
    std::memset(order1_, 0, sizeof(order1_));
    model_order2_.reset();
    model_order3_.reset();
    model_order4_.reset();
    model_order5_.reset();
    model_order6_.reset();
    model_order8_.reset();
    model_sparse1_.reset();
    model_sparse2_.reset();
    model_sparse3_.reset();
    model_word_.reset();
    model_xml_.reset();
    match_model_.reset();

    for (int c = 0; c < 256; ++c) {
        for (int i = 0; i < NUM_MODELS; ++i) weights_[c][i] = 1024;
        weights_[c][13] = 4096;
    }

    sse_.reset();
    std::memset(history_, 0, sizeof(history_));
    hist_pos_ = 0;

    curr_word_hash_ = 0;
    prev_word_hash_ = 0;

    xml_tag_hash_ = 0;
    inside_tag_ = false;
    inside_quotes_ = false;
    bracket_depth_ = 0;
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

    match_model_.add_byte(byte);

    if (byte <= 32 || byte == '"' || byte == '<' || byte == '>' || byte == '[' || byte == ']' || byte == '{' || byte == '}') {
        if (curr_word_hash_ != 0) {
            prev_word_hash_ = curr_word_hash_;
            curr_word_hash_ = 0;
        }
    } else {
        curr_word_hash_ = (curr_word_hash_ * 33ULL) ^ byte;
    }

    if (byte == '<') {
        inside_tag_ = true;
        xml_tag_hash_ = 0;
    } else if (byte == '>') {
        inside_tag_ = false;
        inside_quotes_ = false;
    } else if (inside_tag_) {
        if (byte == '"') inside_quotes_ = !inside_quotes_;
        else if (!inside_quotes_ && byte > 32) xml_tag_hash_ = (xml_tag_hash_ * 33) ^ byte;
    } else {
        if (byte == '[') bracket_depth_++;
        else if (byte == ']' && bracket_depth_ > 0) bracket_depth_--;
    }
}

uint8_t ContextMixer::get_prev(size_t offset) const {
    return history_[(hist_pos_ - offset) & 63];
}

uint16_t ContextMixer::predict_bit(uint8_t c0, uint16_t preds[NUM_MODELS], size_t& sse_bucket) {
    uint8_t b1 = get_prev(1);
    uint8_t b2 = get_prev(2);
    uint8_t b3 = get_prev(3);
    uint8_t b4 = get_prev(4);
    uint8_t b5 = get_prev(5);
    uint8_t b6 = get_prev(6);
    uint8_t b7 = get_prev(7);
    uint8_t b8 = get_prev(8);

    uint64_t h2 = (static_cast<uint64_t>(b2) << 8) | b1;
    uint64_t h3 = (h2 << 8) | b3;
    uint64_t h4 = (h3 << 8) | b4;
    uint64_t h5 = (h4 * 31337ULL) ^ b5;
    uint64_t h6 = (h5 * 31337ULL) ^ b6;
    uint64_t h8 = (h6 * 31337ULL) ^ (static_cast<uint64_t>(b7) << 8) ^ b8;

    uint64_t sp1 = (static_cast<uint64_t>(b3) << 8) | b1; // skip b2
    uint64_t sp2 = (static_cast<uint64_t>(b4) << 8) | b1; // skip b2, b3
    uint64_t sp3 = (static_cast<uint64_t>(b3) << 8) | b2; // skip b1

    uint64_t word_ctx = (prev_word_hash_ * 1000003ULL) ^ curr_word_hash_;
    uint64_t xml_ctx = (static_cast<uint64_t>(xml_tag_hash_) << 8) | 
                       (inside_tag_ ? 1 : 0) | 
                       (inside_quotes_ ? 2 : 0) | 
                       ((bracket_depth_ > 0 ? 1 : 0) << 2);

    preds[0]  = g_mahoney.state_prob[order0_[c0]];
    preds[1]  = g_mahoney.state_prob[order1_[b1][c0]];
    preds[2]  = g_mahoney.state_prob[model_order2_.get(h2, c0)];
    preds[3]  = g_mahoney.state_prob[model_order3_.get(h3, c0)];
    preds[4]  = g_mahoney.state_prob[model_order4_.get(h4, c0)];
    preds[5]  = g_mahoney.state_prob[model_order5_.get(h5, c0)];
    preds[6]  = g_mahoney.state_prob[model_order6_.get(h6, c0)];
    preds[7]  = g_mahoney.state_prob[model_order8_.get(h8, c0)];
    preds[8]  = g_mahoney.state_prob[model_sparse1_.get(sp1, c0)];
    preds[9]  = g_mahoney.state_prob[model_sparse2_.get(sp2, c0)];
    preds[10] = g_mahoney.state_prob[model_sparse3_.get(sp3, c0)];
    preds[11] = g_mahoney.state_prob[model_word_.get(word_ctx, c0)];
    preds[12] = g_mahoney.state_prob[model_xml_.get(xml_ctx, c0)];
    preds[13] = match_model_.predict_bit(c0);

    int32_t sum = 0;
    size_t mlen = match_model_.current_len();
    if (mlen >= 6) {
        int16_t l_match = g_fixed_tables.logit(preds[13]);
        sum = static_cast<int32_t>(l_match) * 3;
    } else {
        for (int k = 0; k < NUM_MODELS; ++k) {
            int16_t l = g_fixed_tables.logit(preds[k]);
            sum += (weights_[c0][k] * static_cast<int32_t>(l)) >> 10;
        }
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
    uint8_t b5 = get_prev(5);
    uint8_t b6 = get_prev(6);
    uint8_t b7 = get_prev(7);
    uint8_t b8 = get_prev(8);

    uint64_t h2 = (static_cast<uint64_t>(b2) << 8) | b1;
    uint64_t h3 = (h2 << 8) | b3;
    uint64_t h4 = (h3 << 8) | b4;
    uint64_t h5 = (h4 * 31337ULL) ^ b5;
    uint64_t h6 = (h5 * 31337ULL) ^ b6;
    uint64_t h8 = (h6 * 31337ULL) ^ (static_cast<uint64_t>(b7) << 8) ^ b8;

    uint64_t sp1 = (static_cast<uint64_t>(b3) << 8) | b1;
    uint64_t sp2 = (static_cast<uint64_t>(b4) << 8) | b1;
    uint64_t sp3 = (static_cast<uint64_t>(b3) << 8) | b2;

    uint64_t word_ctx = (prev_word_hash_ * 1000003ULL) ^ curr_word_hash_;
    uint64_t xml_ctx = (static_cast<uint64_t>(xml_tag_hash_) << 8) | 
                       (inside_tag_ ? 1 : 0) | 
                       (inside_quotes_ ? 2 : 0) | 
                       ((bracket_depth_ > 0 ? 1 : 0) << 2);

    order0_[c0] = g_mahoney.next_state[order0_[c0]][bit];
    order1_[b1][c0] = g_mahoney.next_state[order1_[b1][c0]][bit];
    model_order2_.update(h2, c0, bit);
    model_order3_.update(h3, c0, bit);
    model_order4_.update(h4, c0, bit);
    model_order5_.update(h5, c0, bit);
    model_order6_.update(h6, c0, bit);
    model_order8_.update(h8, c0, bit);
    model_sparse1_.update(sp1, c0, bit);
    model_sparse2_.update(sp2, c0, bit);
    model_sparse3_.update(sp3, c0, bit);
    model_word_.update(word_ctx, c0, bit);
    model_xml_.update(xml_ctx, c0, bit);

    size_t mlen = match_model_.current_len();
    if (mlen < 6) {
        int32_t error = (bit ? PROB_SCALE : 0) - static_cast<int32_t>(p_final);
        for (int k = 0; k < NUM_MODELS; ++k) {
            int16_t l = g_fixed_tables.logit(preds[k]);
            int32_t dw = (error * static_cast<int32_t>(l)) >> 13;
            weights_[c0][k] = std::clamp(weights_[c0][k] + dw, -32768, 32767);
        }
    }
}

std::vector<uint8_t> ContextMixer::compress(const uint8_t* in,
                                            size_t n,
                                            const uint8_t* seed,
                                            size_t seed_len,
                                            ProgressCallback progress) {
    if (n == 0) return {};

    RangeBitEncoder enc;
    ContextMixer mixer;
    if (seed && seed_len > 0) {
        mixer.warm_seed(seed, seed_len);
    }

    uint16_t preds[NUM_MODELS];
    size_t sse_bucket = 0;

    size_t report_interval = std::max<size_t>(1000000, n / 100);

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

        if (progress && (i % report_interval == 0 || i == n - 1)) {
            progress(i + 1, n);
        }
    }
    enc.finish();
    return enc.release_buffer();
}

std::vector<uint8_t> ContextMixer::decompress(const uint8_t* in,
                                              size_t comp_len,
                                              size_t orig_len,
                                              const uint8_t* seed,
                                              size_t seed_len,
                                              ProgressCallback progress) {
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

    size_t report_interval = std::max<size_t>(1000000, orig_len / 100);

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

        if (progress && (i % report_interval == 0 || i == orig_len - 1)) {
            progress(i + 1, orig_len);
        }
    }
    return out;
}

} // namespace pax
