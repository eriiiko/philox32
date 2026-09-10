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
#define PHILOX32_KAT_XLANG_PAIRS 64u
// Number of DRAW lines (the recommended-layout composition, cross-language).
#define PHILOX32_KAT_DRAWS 8u

// Parses lines "philox4x32 10 c0 c1 c2 c3 k0 k1 o0 o1 o2 o3" (hex); skips '#', blanks
// and any other cipher's lines.  The round count is compared against
// PHILOX32_ROUNDS on purpose: change the header's round count and the parser
// finds zero rows -> RESULT: FAIL, so the KAT file and the header cannot drift apart.
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
        unsigned long long v[10];
        bool ok = true;
        for (int i = 0; i < 10; ++i) {
            std::string tok;
            if (!(ss >> tok)) { ok = false; break; }
            char* end = NULL;
            v[i] = std::strtoull(tok.c_str(), &end, 16);
            if (end == tok.c_str() || *end != 0 || v[i] > 0xFFFFFFFFull) { ok = false; break; }
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
// Hand-mirrored by derive_pair() in tests/conftest.py; the mirror is self-gating
// because pytest compares the PAIR ctr/key words printed here against its own
// derivation before comparing outputs (test_cpp_prints_the_agreed_pairs).
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
    std::printf("PAIR %u %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
                (unsigned)i,
                (unsigned)ctr[0], (unsigned)ctr[1], (unsigned)ctr[2], (unsigned)ctr[3],
                (unsigned)key[0], (unsigned)key[1],
                (unsigned)out[0], (unsigned)out[1], (unsigned)out[2], (unsigned)out[3],
                (unsigned)philox32_uniform_q16(out[0]), (unsigned)philox32_uniform_q16(out[1]),
                (unsigned)philox32_uniform_q16(out[2]), (unsigned)philox32_uniform_q16(out[3]),
                (unsigned)philox32_angle_q16(out[0]), (unsigned)philox32_angle_q16(out[1]),
                (unsigned)philox32_angle_q16(out[2]), (unsigned)philox32_angle_q16(out[3]),
                (unsigned)philox32_below(out[0], out[1]), (unsigned)philox32_below(out[1], out[2]),
                (unsigned)philox32_below(out[2], out[3]), (unsigned)philox32_below(out[3], out[0]));
}

// DRAW i seed_lo seed_hi agent_id tick draw_index stream_salt out[4]: the
// recommended layout composed by philox32_draw, so a wire-format drift between
// the C++ and Python helpers (which word agent_id lands in) is caught even
// though the block function itself already agrees.  Inputs mirror
// derive_draw_inputs() in tests/conftest.py.
inline void derive_draw_inputs(uint32_t i, uint32_t in[6])
{
    for (uint32_t j = 0; j < 6; ++j)
        in[j] = (uint32_t)(0x9E3779B9u * (i + 1u) + 0x85EBCA6Bu * j) ^ (i < 4u ? 0u : 0xFFFFFFFFu);
}

inline void print_draw(uint32_t i, const uint32_t in[6])
{
    uint32_t out[4];
    philox32_draw(in[0], in[1], in[2], in[3], in[4], in[5], out);
    std::printf("DRAW %u %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n", (unsigned)i,
                (unsigned)in[0], (unsigned)in[1], (unsigned)in[2], (unsigned)in[3], (unsigned)in[4], (unsigned)in[5],
                (unsigned)out[0], (unsigned)out[1], (unsigned)out[2], (unsigned)out[3]);
}

#endif // PHILOX32_KAT_COMMON_H
