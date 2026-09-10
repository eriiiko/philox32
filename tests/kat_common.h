// kat_common.h -- shared helpers for the host and device KAT checkers.
// KAT file parsing, the deterministic derivation of the 64 cross-language pairs,
// and the PAIR line printer.  Host-only code (not compiled for device).
//
// License: MIT, Copyright (c) 2026 Erik Steen.
#ifndef PHILOX32_KAT_COMMON_H
#define PHILOX32_KAT_COMMON_H

#include "philox32.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct KatRow {
    uint32_t ctr[4];
    uint32_t key[2];
    uint32_t out[4];
};

// Number of cross-language pairs printed as PAIR lines.
#define XLANG_PAIRS 64u

// Parses lines "philox4x32 10 c0 c1 c2 c3 k0 k1 o0 o1 o2 o3" (hex); skips '#' and blanks.
inline bool load_kat(const char* path, std::vector<KatRow>& rows)
{
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string name;
        unsigned rounds = 0;
        ss >> name >> rounds;
        if (name != "philox4x32" || rounds != PHILOX32_ROUNDS) continue;
        KatRow r;
        unsigned long v[10];
        bool ok = true;
        for (int i = 0; i < 10; ++i) {
            std::string tok;
            if (!(ss >> tok)) { ok = false; break; }
            v[i] = std::strtoul(tok.c_str(), 0, 16);
        }
        if (!ok) {
            std::fprintf(stderr, "malformed KAT line: %s\n", line.c_str());
            return false;
        }
        for (int i = 0; i < 4; ++i) r.ctr[i] = (uint32_t)v[i];
        for (int i = 0; i < 2; ++i) r.key[i] = (uint32_t)v[4 + i];
        for (int i = 0; i < 4; ++i) r.out[i] = (uint32_t)v[6 + i];
        rows.push_back(r);
    }
    return true;
}

// Pair i is built from KAT row (i mod n): its published OUTPUT words become the
// counter and its key words the key, each perturbed by a distinct Weyl-style
// stride so the 64 pairs are all different and spread across the 32-bit space.
// Mirrored exactly by derive_pair() in tests/conftest.py.
inline void derive_pair(const std::vector<KatRow>& rows, uint32_t i, uint32_t ctr[4], uint32_t key[2])
{
    const KatRow& base = rows[i % rows.size()];
    for (uint32_t j = 0; j < 4; ++j)
        ctr[j] = base.out[j] ^ (uint32_t)(i * 0x9E3779B9u + j * 0x85EBCA6Bu);
    for (uint32_t j = 0; j < 2; ++j)
        key[j] = base.key[j] + (uint32_t)(i * 0xC2B2AE35u + j * 0x27D4EB2Fu);
}

inline void print_pair(uint32_t i, const uint32_t ctr[4], const uint32_t key[2], const uint32_t out[4])
{
    std::printf("PAIR %u %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
                (unsigned)i,
                (unsigned)ctr[0], (unsigned)ctr[1], (unsigned)ctr[2], (unsigned)ctr[3],
                (unsigned)key[0], (unsigned)key[1],
                (unsigned)out[0], (unsigned)out[1], (unsigned)out[2], (unsigned)out[3],
                (unsigned)philox32_uniform_q16(out[0]), (unsigned)philox32_uniform_q16(out[1]),
                (unsigned)philox32_uniform_q16(out[2]), (unsigned)philox32_uniform_q16(out[3]),
                (unsigned)philox32_angle_q16(out[0]), (unsigned)philox32_angle_q16(out[1]),
                (unsigned)philox32_angle_q16(out[2]), (unsigned)philox32_angle_q16(out[3]));
}

#endif // PHILOX32_KAT_COMMON_H
