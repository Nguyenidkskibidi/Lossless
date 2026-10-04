#include "pax/bwt.hpp"
#include <algorithm>
#include <thread>
#include <future>

namespace pax {

std::pair<std::vector<uint8_t>, uint32_t> BWT::encode(const uint8_t* in, size_t n, int /*num_threads*/) {
    if (n == 0) return {{}, 0};
    if (n == 1) return {{in[0]}, 0};

    // Fast check: if all bytes are identical, BWT is identity and primary is 0
    bool all_same = true;
    for (size_t i = 1; i < n; ++i) {
        if (in[i] != in[0]) {
            all_same = false;
            break;
        }
    }
    if (all_same) {
        return {std::vector<uint8_t>(in, in + n), 0};
    }

    // Prefix Doubling (Manber-Myers / Larsson-Sadakane) Suffix Sort for cyclic BWT
    std::vector<int> sa(n);
    std::vector<int> rank(n);
    std::vector<int> next_rank(n);

    for (size_t i = 0; i < n; ++i) {
        sa[i] = static_cast<int>(i);
        rank[i] = in[i];
    }

    size_t count_size = std::max<size_t>(256, n) + 1;
    std::vector<int> count(count_size, 0);
    std::vector<int> tmp_sa(n);

    for (int k = 1; k < static_cast<int>(n); k <<= 1) {
        // 1. Sort by second key: rank[(i + k) % n]
        std::fill(count.begin(), count.end(), 0);
        for (size_t i = 0; i < n; ++i) {
            count[rank[(i + k) % n]]++;
        }
        for (size_t i = 1; i < count_size; ++i) {
            count[i] += count[i - 1];
        }
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            int second_val = rank[(i + k) % n];
            tmp_sa[--count[second_val]] = i;
        }

        // 2. Sort by first key: rank[tmp_sa[i]]
        std::fill(count.begin(), count.end(), 0);
        for (size_t i = 0; i < n; ++i) {
            count[rank[tmp_sa[i]]]++;
        }
        for (size_t i = 1; i < count_size; ++i) {
            count[i] += count[i - 1];
        }
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            int first_val = rank[tmp_sa[i]];
            sa[--count[first_val]] = tmp_sa[i];
        }

        // 3. Recompute ranks
        next_rank[sa[0]] = 0;
        int r = 0;
        for (size_t i = 1; i < n; ++i) {
            int prev = sa[i - 1];
            int curr = sa[i];
            if (rank[prev] != rank[curr] || rank[(prev + k) % n] != rank[(curr + k) % n]) {
                r++;
            }
            next_rank[curr] = r;
        }
        rank = next_rank;
        if (r == static_cast<int>(n) - 1) break; // All suffixes uniquely identified
    }

    std::vector<uint8_t> L(n);
    uint32_t primary = 0;
    for (size_t i = 0; i < n; ++i) {
        if (sa[i] == 0) primary = static_cast<uint32_t>(i);
        L[i] = in[(sa[i] + n - 1) % n];
    }
    return {std::move(L), primary};
}

std::vector<uint8_t> BWT::decode(const uint8_t* L, size_t n, uint32_t primary_index) {
    if (n == 0) return {};
    if (n == 1) return {L[0]};

    // 1. Symbol frequency counting
    std::vector<uint32_t> C(256, 0);
    for (size_t i = 0; i < n; ++i) C[L[i]]++;

    // 2. Prefix sums of frequencies
    std::vector<uint32_t> P(256, 0);
    uint32_t sum = 0;
    for (int i = 0; i < 256; ++i) {
        P[i] = sum;
        sum += C[i];
    }

    // 3. LF-mapping table construction
    std::vector<uint32_t> LF(n);
    for (size_t i = 0; i < n; ++i) {
        LF[i] = P[L[i]]++;
    }

    // 4. Backward walk from primary index
    std::vector<uint8_t> out(n);
    uint32_t curr = primary_index;
    for (size_t i = n; i > 0; --i) {
        out[i - 1] = L[curr];
        curr = LF[curr];
    }
    return out;
}

} // namespace pax
