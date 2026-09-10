// kat_check.cpp -- host-side known-answer test + cross-language dump for philox32.h.
//
// Usage:  kat_check <kat_philox4x32_10.txt>
// Output protocol (parsed by tests/test_cpp_kat.py):
//   KAT <passed>/<total>
//   PAIR <i> c0 c1 c2 c3 k0 k1 o0 o1 o2 o3 u0 u1 u2 u3 a0 a1 a2 a3   (64 lines, hex)
//   RESULT: PASS | RESULT: FAIL
// The 64 PAIR inputs are derived deterministically from the KAT vectors (see
// derive_pair, mirrored in tests/conftest.py); o = philox output words,
// u = philox32_uniform_q16(o), a = philox32_angle_q16(o).
//
// License: MIT, Copyright (c) 2026 Erik Steen.
#include "philox32.h"
#include "kat_common.h"

#include <cstdio>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: kat_check <kat file>\n");
        return 2;
    }
    std::vector<KatRow> rows;
    if (!load_kat(argv[1], rows) || rows.empty()) {
        std::printf("RESULT: FAIL\n");
        return 1;
    }

    unsigned passed = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        uint32_t out[4];
        philox32_4x32_10(rows[i].ctr, rows[i].key, out);
        if (out[0] == rows[i].out[0] && out[1] == rows[i].out[1] &&
            out[2] == rows[i].out[2] && out[3] == rows[i].out[3]) {
            ++passed;
        }
    }
    std::printf("KAT %u/%u\n", passed, (unsigned)rows.size());

    for (uint32_t i = 0; i < XLANG_PAIRS; ++i) {
        uint32_t ctr[4], key[2], out[4];
        derive_pair(rows, i, ctr, key);
        philox32_4x32_10(ctr, key, out);
        print_pair(i, ctr, key, out);
    }

    const bool ok = (passed == rows.size());
    std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
