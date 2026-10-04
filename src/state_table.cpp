#include "pax/state_table.hpp"

namespace pax {

MahoneyAutomaton::MahoneyAutomaton() {
    for (int s = 0; s < 256; ++s) {
        next_state[s][0] = MAHONEY_STATE_TABLE[s][0];
        next_state[s][1] = MAHONEY_STATE_TABLE[s][1];

        int n0 = MAHONEY_STATE_TABLE[s][2];
        int n1 = MAHONEY_STATE_TABLE[s][3];
        if (s == 0) {
            state_prob[s] = 2048;
        } else {
            double p = (n1 + 0.5) / (n0 + n1 + 1.0);
            int p_scaled = static_cast<int>(std::round(p * 4096.0));
            state_prob[s] = static_cast<uint16_t>(std::clamp(p_scaled, 1, 4095));
        }
    }
}

const MahoneyAutomaton g_mahoney;

} // namespace pax
