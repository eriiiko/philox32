// philox32.h -- Philox-4x32-10, header-only, integer-only, host + CUDA device.
//
// Algorithm: John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw,
//   "Parallel random numbers: as easy as 1, 2, 3", Proceedings of SC'11,
//   DOI 10.1145/2063384.2063405.  Reference implementation: Random123
//   (https://github.com/DEShawResearch/random123, BSD-3-Clause).  This file is
//   an independent re-implementation gated against Random123's published
//   known-answer vectors (tests/kat_philox4x32_10.txt).
//
// License: MIT, Copyright (c) 2026 Erik Steen.  See LICENSE.
//
// WHAT
//   Philox is a counter-based, stateless, keyed RNG: out = F(ctr, key).  There
//   is no internal state to seed, advance or share -- the caller names the draw
//   it wants (by counter + key) and gets the same 128 bits on every platform.
//   Only uint32_t / uint64_t arithmetic is used (32x32->64 products via a
//   uint64_t multiply), so host and device agree bit for bit, and so does the
//   pure-Python twin in philox32.py.
//
// API
//   philox32_4x32_10(ctr[4], key[2], out[4])   the block function, 10 rounds
//   philox32_uniform_q16(w)  -> Q16.16 in [0, 1)        == w >> 16
//   philox32_angle_q16(w)    -> Q16.16 in [0, 2*pi)     == (w * TWO_PI_Q16) >> 32
//   philox32_make_key / philox32_make_counter   the recommended layout below
//
// RECOMMENDED KEY / COUNTER LAYOUT FOR AGENT SIMULATIONS
//   key     = (seed_lo, seed_hi ^ agent_id)
//   counter = (tick, draw_index, stream_salt, 0)
//   - seed is the simulation's 64-bit seed; agent_id makes every agent an
//     independent stream; stream_salt separates purposes within one agent and
//     tick (e.g. movement vs combat) so adding a new consumer never shifts the
//     draws of an existing one; draw_index counts the draws a consumer makes
//     within one tick.  Each block yields four 32-bit words -- use out[0..3]
//     before bumping draw_index.
//
// DETERMINISM NOTES
//   - No floating point anywhere: the mappings to Q16.16 are integer shifts and
//     one 64-bit product.  TWO_PI_Q16 = 411775 = round(2*pi * 65536) is a
//     checked-in constant, never computed at run time.
//   - Outputs are pure functions of (ctr, key); the same inputs give the same
//     bits on MSVC, GCC, Clang, nvcc device code, CPython and numpy.
#ifndef PHILOX32_H
#define PHILOX32_H

#include <stdint.h>

#if defined(__CUDACC__)
#define PHILOX32_HD __host__ __device__
#else
#define PHILOX32_HD
#endif

// Round multipliers and Weyl key increments (Random123 PHILOX_M4x32_* / PHILOX_W32_*).
#define PHILOX32_M0 0xD2511F53u
#define PHILOX32_M1 0xCD9E8D57u
#define PHILOX32_W0 0x9E3779B9u
#define PHILOX32_W1 0xBB67AE85u
#define PHILOX32_ROUNDS 10

// round(2*pi * 65536): 2*pi in Q16.16.
#define PHILOX32_TWO_PI_Q16 411775u

// One Philox round.  hi/lo of 32x32->64 products, then the permute + key xor.
PHILOX32_HD inline void philox32_round(uint32_t ctr[4], const uint32_t key[2])
{
    const uint64_t p0 = (uint64_t)PHILOX32_M0 * (uint64_t)ctr[0];
    const uint64_t p1 = (uint64_t)PHILOX32_M1 * (uint64_t)ctr[2];
    const uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
    const uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
    const uint32_t c1 = ctr[1], c3 = ctr[3];
    ctr[0] = hi1 ^ c1 ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ c3 ^ key[1];
    ctr[3] = lo0;
}

// Philox-4x32-10: round 1 with the raw key, then (bump key, round) x 9.
PHILOX32_HD inline void philox32_4x32_10(const uint32_t ctr[4], const uint32_t key[2], uint32_t out[4])
{
    uint32_t c[4] = { ctr[0], ctr[1], ctr[2], ctr[3] };
    uint32_t k[2] = { key[0], key[1] };
    philox32_round(c, k);
    for (int r = 1; r < PHILOX32_ROUNDS; ++r) {
        k[0] += PHILOX32_W0;   // uint32_t wrap-around is well defined
        k[1] += PHILOX32_W1;
        philox32_round(c, k);
    }
    out[0] = c[0]; out[1] = c[1]; out[2] = c[2]; out[3] = c[3];
}

// Uniform in [0, 1) as Q16.16 (65536 == 1.0): the top 16 bits of w.
// Range: 0 .. 65535 inclusive.
PHILOX32_HD inline uint32_t philox32_uniform_q16(uint32_t w)
{
    return w >> 16;
}

// Uniform angle in [0, 2*pi) as Q16.16: floor(w * 2pi_q16 / 2^32).
// Range: 0 .. PHILOX32_TWO_PI_Q16 - 1 inclusive.
PHILOX32_HD inline uint32_t philox32_angle_q16(uint32_t w)
{
    return (uint32_t)(((uint64_t)w * (uint64_t)PHILOX32_TWO_PI_Q16) >> 32);
}

// The recommended layout, made executable.
PHILOX32_HD inline void philox32_make_key(uint32_t seed_lo, uint32_t seed_hi, uint32_t agent_id, uint32_t key[2])
{
    key[0] = seed_lo;
    key[1] = seed_hi ^ agent_id;
}

PHILOX32_HD inline void philox32_make_counter(uint32_t tick, uint32_t draw_index, uint32_t stream_salt, uint32_t ctr[4])
{
    ctr[0] = tick;
    ctr[1] = draw_index;
    ctr[2] = stream_salt;
    ctr[3] = 0u;
}

#endif // PHILOX32_H
