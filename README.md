# philox32

Philox-4x32-10 -- the counter-based random number generator of Salmon, Moraes,
Dror and Shaw (SC'11) -- as one header for C++ and CUDA and one Python module
that produce **bit-identical** draws.  Integer-only, zero dependencies, MIT.

A counter-based RNG has no internal state to seed and advance: you hand it a
*counter* (which draw you want) and a *key* (whose stream it belongs to) and it
returns 128 pseudo-random bits, the same bits on every platform, every time.
For deterministic simulations that changes everything.  A CPU implementation,
a GPU kernel and a Python analysis script can each compute "agent 17's third
movement draw on tick 4021" independently and agree to the bit, with no shared
state, no locks, and no order dependence between threads.  That is what makes
replays exact, lockstep multiplayer stay in sync from the seed alone, and RL
experiments reproducible across a laptop, a cluster and a notebook.  philox32
gives you that with plain `uint32_t`/`uint64_t` arithmetic -- no floating point
anywhere -- and a test suite that proves the three implementations agree
against the algorithm's published known-answer vectors.

Use it however you like -- it is MIT licensed.  Issues and pull requests are
welcome, whether it is a port to another language, a new output mapping, or a
platform where the bits come out different (that would be a bug we want to know
about).

It grew out of two deterministic-simulation projects (breach, civulator) that
needed CPU code, CUDA kernels and Python tooling to share one RNG.

## Install / vendor

There is nothing to build.  Copy `philox32.h` next to your sources (C++ or
CUDA), or copy `philox32.py` into your project.  For Python you can also
`pip install .` from a checkout (`pip install ".[numpy]"` adds the optional
vectorised path).

## Citation

John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw.
*Parallel random numbers: as easy as 1, 2, 3.*  SC'11.
DOI [10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405).

Reference implementation: [Random123](https://github.com/DEShawResearch/random123)
(D. E. Shaw Research, BSD-3-Clause).  This package is an independent
re-implementation; its correctness is gated by Random123's published
known-answer vectors (`tests/kat_philox4x32_10.txt`, see
`THIRD_PARTY_NOTICES.md`).

## How it works

`out[4] = F(ctr[4], key[2])`: a 128-bit counter, a 64-bit key, ten rounds.
Per round: `hi0,lo0 = 0xD2511F53 * ctr[0]`, `hi1,lo1 = 0xCD9E8D57 * ctr[2]`
(32x32 -> 64), then `ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]`.
Round 1 uses the raw key; each of the following 9 rounds first bumps
`key[0] += 0x9E3779B9`, `key[1] += 0xBB67AE85`.  Only multiply, xor, add and
shift on unsigned integers -- nothing that could round differently anywhere.

## Files

| File | What |
|---|---|
| `philox32.h` | header-only C++11; `PHILOX32_HD` = `__host__ __device__` under nvcc |
| `philox32.py` | pure-Python twin + a vectorised numpy variant (uint32/uint64 only) |
| `tests/kat_philox4x32_10.txt` | Random123's published vectors for philox4x32-10 |
| `tests/kat_check.cpp` | host KAT + cross-language dump (`RESULT: PASS/FAIL`) |
| `tests/kat_check_device.cu` | the same on the GPU, plus host-vs-device identity |
| `tests/test_*.py` | pytest: Python KAT, C++ KAT (builds with MSVC), CUDA KAT (nvcc), cross-language identity, Q16 mappings |

## Usage

### C++ / CUDA

```cpp
#include "philox32.h"

uint32_t key[2], ctr[4], out[4];
philox32_make_key(seed_lo, seed_hi, agent_id, key);      // (seed_lo, seed_hi ^ agent_id)
philox32_make_counter(tick, draw_index, stream_salt, ctr); // (tick, draw_index, stream_salt, 0)
philox32_4x32_10(ctr, key, out);                          // four 32-bit words

uint32_t u = philox32_uniform_q16(out[0]);   // Q16.16 in [0, 1):    0 .. 65535
uint32_t a = philox32_angle_q16(out[1]);     // Q16.16 in [0, 2*pi): 0 .. 411774
```

Every function is `PHILOX32_HD`, so the same header compiles into `__global__`
kernels unchanged -- one thread per draw, no state, no atomics.

### Python

```python
import philox32 as px

key = px.philox32_make_key(seed_lo, seed_hi, agent_id)
ctr = px.philox32_make_counter(tick, draw_index, stream_salt)
out = px.philox32_4x32_10(ctr, key)          # tuple of four ints in [0, 2**32)
u = px.philox32_uniform_q16(out[0])          # 0 .. 65535
a = px.philox32_angle_q16(out[1])            # 0 .. 411774

# vectorised (numpy, exact): ctr uint32 (n, 4), key uint32 (n, 2) -> uint32 (n, 4)
outs = px.philox32_4x32_10_np(ctr_arr, key_arr)
us = px.philox32_uniform_q16_np(outs[:, 0])
```

The scalar twin rejects anything that is not an int in `[0, 2**32)`; the numpy
variant rejects anything that is not already `uint32`.  Both refuse to guess.

## Recommended key / counter layout for agent simulations

```
key     = (seed_lo, seed_hi ^ agent_id)
counter = (tick, draw_index, stream_salt, 0)
```

- `seed` -- the simulation's 64-bit seed, split into two words.
- `agent_id` -- xored into the high seed word: every agent is an independent
  stream, with no allocation or bookkeeping.
- `tick` -- the simulation step; draws never depend on what happened earlier.
- `stream_salt` -- one constant per *purpose* (movement, combat, spawn...).
  Adding a new consumer gets a new salt and never shifts an existing one's draws.
- `draw_index` -- the n-th block a consumer takes within one tick.  Each block
  yields four words; use all four before bumping the index.
- word 3 is reserved (0); a future need for more counter space has a home.

## Determinism notes

- **No floats anywhere.**  The block function is integer-only; the Q16.16
  mappings are a shift (`w >> 16`) and one 64-bit product
  (`(w * 411775) >> 32`).  `TWO_PI_Q16 = 411775 = round(2*pi * 65536)` is a
  checked-in constant, never computed at run time.
- **Outputs are pure functions of the inputs.**  Same `(ctr, key)` -> same bits
  on MSVC, GCC, Clang, nvcc device code, CPython and numpy.  The tests prove
  this on every run: the KAT vectors pin the algorithm, `kat_check` dumps 64
  derived pairs that pytest compares bit-for-bit with the Python twin, and the
  device build compares GPU output against the host header inside one binary.
- The numpy variant uses explicit `uint32`/`uint64` dtypes and the natural
  wrap-around; it never falls back to Python ints or floats.

## Tests

```
pytest -q
```

`tests/conftest.py` locates MSVC through `vswhere` (preferring VS 2022, the
host compiler nvcc 12.x supports) and builds `tests/kat_check.cpp` with
`cl /W4 /WX /O2 /std:c++14` into `build/`; `kat_check_device.cu` is built with
`nvcc -arch=sm_75` when nvcc is present.  Both tests skip with a stated reason
when the toolchain is missing, and the device test also skips when no CUDA
device is usable.  On a machine without MSVC, build by hand, e.g.
`g++ -std=c++11 -O2 -I. -Itests tests/kat_check.cpp -o build/kat_check && build/kat_check tests/kat_philox4x32_10.txt`.

## License

MIT, Copyright (c) 2026 Erik Steen.  Third-party material: `THIRD_PARTY_NOTICES.md`.
