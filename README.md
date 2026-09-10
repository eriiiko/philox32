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
anywhere -- and a test suite that checks the implementations against each
other and against the algorithm's published known-answer vectors.

Use it however you like -- it is MIT licensed.  Issues and pull requests are
welcome, whether it is a port to another language, a new output mapping, or a
platform where the bits come out different (that would be a bug we want to know
about).

It grew out of two deterministic-simulation projects (breach, civulator) that
needed CPU code, CUDA kernels and Python tooling to share one RNG.

## Install / vendor

There is nothing to build.  Copy `philox32/philox32.h` next to your sources
(C++ or CUDA), or copy `philox32/philox32.py` into your project.  For Python,
`pip install .` from a checkout installs the package *and* the header:
`philox32.header_path()` returns the installed `philox32.h`, and
`philox32.header_dir()` is what to pass as `-I`.  `pip install ".[numpy]"` adds
the optional vectorised path.

## Citation

John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw.
*Parallel random numbers: as easy as 1, 2, 3.*  SC'11.
DOI [10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405).

Reference implementation: [Random123](https://github.com/DEShawResearch/random123)
(D. E. Shaw Research, BSD-3-Clause).  This package is an independent
re-implementation; its correctness is gated by Random123's published
known-answer vectors (`tests/kat_philox4x32_10.txt`, see
`THIRD_PARTY_NOTICES.md`) and, where a GPU is present, by NVIDIA cuRAND's
independent `curand_Philox4x32_10` on 200 000 fuzzed inputs.

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
| `philox32/philox32.h` | header-only C++11 (the library; the test harness itself needs C++14 on MSVC); `PHILOX32_HD` = `__host__ __device__` under nvcc / hipcc |
| `philox32/philox32.py` | pure-Python twin + a vectorised numpy variant (uint32/uint64 only) |
| `philox32/__init__.py` | re-exports the twin; `header_path()` / `header_dir()` |
| `tests/kat_philox4x32_10.txt` | Random123's published vectors for philox4x32-10 |
| `tests/kat_check.cpp` | host KAT + cross-language dump (`RESULT: PASS/FAIL`) |
| `tests/kat_check_device.cu` | the same on the GPU, plus host-vs-device identity |
| `tests/curand_oracle.cu` | philox32 vs cuRAND's independent Philox on 200 000 fuzzed pairs + 12 edge rows |
| `tests/test_*.py` | pytest: Python KAT, C++ KAT (builds with the platform compiler), CUDA KAT + oracle (nvcc), cross-language identity, mappings, layout, packaging |
| `VERIFIED.md` | dated device-run records (GPU runs cannot happen in CI) |

## Usage

### C++ / CUDA

```cpp
#include "philox32.h"

// The recommended layout in one call:
uint32_t out[4];
philox32_draw(seed_lo, seed_hi, agent_id, tick, draw_index, SALT_MOVEMENT, out);

// or explicitly:
uint32_t key[2], ctr[4];
philox32_make_key(seed_lo, seed_hi, key);                            // (seed_lo, seed_hi)
philox32_make_counter(tick, draw_index, SALT_MOVEMENT, agent_id, ctr); // (tick, draw_index, salt, agent_id)
philox32_4x32_10(ctr, key, out);                                     // four 32-bit words

uint32_t u = philox32_uniform_q16(out[0]);   // Q16.16 in [0, 1):    0 .. 65535
uint32_t a = philox32_angle_q16(out[1]);     // Q16.16 in [0, 2*pi): 0 .. 411774
uint32_t k = philox32_below(out[2], 6);      // integer in [0, 6)
```

Every function is `PHILOX32_HD`, so the same header compiles into `__global__`
kernels unchanged -- one thread per draw, no state, no atomics.  Predefine
`PHILOX32_HD` before including to use your own annotation.

### Python

```python
import philox32 as px

out = px.philox32_draw(seed_lo, seed_hi, agent_id, tick, draw_index, SALT_MOVEMENT)
u = px.philox32_uniform_q16(out[0])          # 0 .. 65535
a = px.philox32_angle_q16(out[1])            # 0 .. 411774
k = px.philox32_below(out[2], 6)             # 0 .. 5

# vectorised (numpy, exact): ctr uint32 (n, 4), key uint32 (n, 2) -> uint32 (n, 4)
outs = px.philox32_4x32_10_np(ctr_arr, key_arr)
us = px.philox32_uniform_q16_np(outs[:, 0])
ks = px.philox32_below_np(outs[:, 2], np.uint32(6))
```

Input contracts differ by language, deliberately: the **C++ helpers take
`uint32_t` and truncate silently** (there is nothing else a header can do); the
**Python scalar functions are the strict twin** -- they accept any int-like
value (`operator.index`: Python ints, numpy integers) and reject bools, floats
and anything outside `[0, 2**32)`.  The numpy variants accept only `uint32`
arrays, so a float or an oversized int can never slip in by promotion.

## Output mappings

All three are integer-only: a shift, or one 64-bit product and a shift.

| Function | Result | Exactness |
|---|---|---|
| `philox32_uniform_q16(w)` = `w >> 16` | Q16.16 in [0, 1), 65536 = 1.0 | exactly uniform; **65536 attainable values** is also the resolution -- take the raw word for anything finer |
| `philox32_angle_q16(w)` = `(w * 411775) >> 32` | Q16.16 in [0, 2*pi) | each value owns 10430 or 10431 of the 2^32 words: max relative excess **9.6e-5** |
| `philox32_below(w, n)` = `(w * n) >> 32` | integer in [0, n) | Lemire multiply-shift, no division; relative excess of the fattest bin **< n / 2^32** (2.3e-9 for n = 10, 2.3e-4 for n = 1e6) |

`TWO_PI_Q16 = 411775 = round(2*pi * 65536)` is a checked-in constant.

There is deliberately **no rejection-sampling "exact" variant** of `below`:
rejection makes the number of words consumed depend on the draw, so two
consumers' `draw_index` bookkeeping would diverge on data -- a determinism
hazard in a lockstep simulation.  If the bias matters, draw two words and
reduce a 64-bit product, or keep `n` small.  `w % n` is worse on every axis
(bias up to 1 part in 2^32/n *and* a division) -- do not reach for it.

## Recommended key / counter layout for agent simulations

```
key     = (seed_lo, seed_hi)                        -- the seed and nothing else
counter = (tick, draw_index, stream_salt, agent_id)
```

- `seed` -- the simulation's 64-bit seed, split into two words.  It is the
  whole key, so two seeds never share a stream for any agent/tick/salt/index;
  an 8 seeds x 8 agents sweep gives 64 distinct streams (gated by
  `tests/test_layout.py`).
- `agent_id` -- counter word 3: every agent is an independent stream within a
  seed, with no allocation or bookkeeping.
- `tick` -- the simulation step; draws never depend on what happened earlier.
- `stream_salt` -- one constant per *purpose* (movement, combat, spawn...).
  Adding a new consumer gets a new salt and never shifts an existing one's
  draws.  **Keep the salts in one enum per consuming project** -- an
  un-namespaced magic number per call site is how two consumers end up sharing
  a salt and, silently, the same draws.
- `draw_index` -- the n-th block a consumer takes within one tick.  Each block
  yields four words; use all four before bumping the index.  Two code paths
  calling with the same `(tick, salt, agent_id, draw_index)` get identical
  words, and nothing can detect that for you -- the salt enum is the defence.

This layout is a **wire format**: once replays or golden digests are recorded
against it, changing it invalidates them.  It was fixed at 0.1.0.

## Determinism notes

- **No floats anywhere.**  The block function is integer-only; the output
  mappings are one shift or one 64-bit product each.
- **Outputs are pure functions of the inputs.**  The block function, the three
  mappings and the layout composition are compared bit-for-bit between the
  C++ header, the CUDA device build and the Python twin on every test run.
- **What has actually been verified** (see `VERIFIED.md`): MSVC 19.4x x64 host
  code, nvcc 12.4 device code on sm_86, CPython 3.11 and numpy 2.x agree with
  each other, with Random123's vectors, and with cuRAND's independent
  implementation on 200 000 fuzzed inputs.  GCC and Clang are *expected* to
  agree -- the header contains no implementation-defined construct (unsigned
  arithmetic, explicit `uint64_t` widening, no signed shifts) -- and are gated
  by the CI matrix in `.github/workflows/ci.yml` (g++/clang++ at C++11 and
  C++17 with `-Werror -pedantic`, plus a UBSan/ASan run); that claim becomes a
  proof only when that CI is green.
- The numpy variant uses explicit `uint32`/`uint64` dtypes and the natural
  wrap-around; it never falls back to Python ints or floats.

## Tests

```
pytest -q
```

`tests/conftest.py` builds `tests/kat_check.cpp` with the platform compiler:
MSVC via `vswhere` on Windows (`cl /W4 /WX /O2 /std:c++14`, preferring VS
2022, the host compiler nvcc 12.x supports), otherwise `$CXX` / `c++` / `g++`
/ `clang++` with `-std=c++11 -O2 -Wall -Wextra -Werror -pedantic`.  When nvcc
is present it also builds `kat_check_device.cu` and `curand_oracle.cu`
(`-arch=native`, falling back to `sm_75`) and runs them on the GPU.  Each leg
skips with a stated reason only when its toolchain is absent (or, for the
device legs, when there is no CUDA device at all -- any other CUDA error is a
failure, not a skip).  Build outputs land in `build/`.

## License

MIT, Copyright (c) 2026 Erik Steen.  Third-party material: `THIRD_PARTY_NOTICES.md`.
