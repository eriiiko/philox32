"""philox32 -- Philox-4x32-10, pure-Python twin of philox32.h (plus a numpy variant).

Algorithm: Salmon, Moraes, Dror, Shaw, "Parallel random numbers: as easy as
1, 2, 3", SC'11, DOI 10.1145/2063384.2063405.  Reference implementation:
Random123 (BSD-3-Clause); this module is gated against its published
known-answer vectors (tests/kat_philox4x32_10.txt).

License: MIT, Copyright (c) 2026 Erik Steen.

Everything here is integer arithmetic on values reduced to 32 bits, so the
results are bit-identical to the C++/CUDA header.  The numpy variant uses
explicit uint32/uint64 dtypes throughout (no Python-int fallback, no float).

Recommended layout for agent simulations (see philox32.h for the reasoning):
    key     = (seed_lo, seed_hi ^ agent_id)
    counter = (tick, draw_index, stream_salt, 0)
"""

__version__ = "0.1.0"

MASK32 = 0xFFFFFFFF

# Round multipliers and Weyl key increments (Random123 PHILOX_M4x32_* / PHILOX_W32_*).
M0 = 0xD2511F53
M1 = 0xCD9E8D57
W0 = 0x9E3779B9
W1 = 0xBB67AE85
ROUNDS = 10

# round(2*pi * 65536): 2*pi in Q16.16.  A checked-in constant, never computed.
TWO_PI_Q16 = 411775


def _check_u32(name, values, n):
    if len(values) != n:
        raise ValueError(f"{name} must have {n} words, got {len(values)}")
    for v in values:
        if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= MASK32):
            raise ValueError(f"{name} words must be ints in [0, 2**32), got {v!r}")


def philox32_4x32_10(ctr, key):
    """Philox-4x32-10 block function.  ctr: 4 ints, key: 2 ints, all in [0, 2**32).

    Returns a tuple of four ints in [0, 2**32).  Pure function of its inputs.
    """
    _check_u32("ctr", ctr, 4)
    _check_u32("key", key, 2)
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in range(ROUNDS):
        if r > 0:
            k0 = (k0 + W0) & MASK32
            k1 = (k1 + W1) & MASK32
        p0 = M0 * c0
        p1 = M1 * c2
        hi0, lo0 = p0 >> 32, p0 & MASK32
        hi1, lo1 = p1 >> 32, p1 & MASK32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return (c0, c1, c2, c3)


def philox32_uniform_q16(w):
    """Uniform in [0, 1) as Q16.16 (65536 == 1.0): the top 16 bits of w.  Range 0..65535."""
    _check_u32("w", (w,), 1)
    return w >> 16


def philox32_angle_q16(w):
    """Uniform angle in [0, 2*pi) as Q16.16: (w * TWO_PI_Q16) >> 32.  Range 0..TWO_PI_Q16-1."""
    _check_u32("w", (w,), 1)
    return (w * TWO_PI_Q16) >> 32


def philox32_make_key(seed_lo, seed_hi, agent_id):
    """The recommended key layout: (seed_lo, seed_hi ^ agent_id)."""
    _check_u32("key inputs", (seed_lo, seed_hi, agent_id), 3)
    return (seed_lo, seed_hi ^ agent_id)


def philox32_make_counter(tick, draw_index, stream_salt):
    """The recommended counter layout: (tick, draw_index, stream_salt, 0)."""
    _check_u32("counter inputs", (tick, draw_index, stream_salt), 3)
    return (tick, draw_index, stream_salt, 0)


# --------------------------------------------------------------------------- numpy
def philox32_4x32_10_np(ctr, key):
    """Vectorised Philox-4x32-10.  ctr: uint32 (n, 4), key: uint32 (n, 2) -> uint32 (n, 4).

    Exact: products are formed in uint64, everything else in uint32 with the
    natural wrap-around.  Both inputs must already be uint32 -- no implicit
    conversion, so a caller can never feed floats or oversized ints by accident.
    """
    import numpy as np

    ctr = np.asarray(ctr)
    key = np.asarray(key)
    if ctr.dtype != np.uint32 or key.dtype != np.uint32:
        raise TypeError(f"ctr and key must be uint32 arrays, got {ctr.dtype} / {key.dtype}")
    if ctr.ndim != 2 or ctr.shape[1] != 4 or key.ndim != 2 or key.shape[1] != 2 or ctr.shape[0] != key.shape[0]:
        raise ValueError(f"expected ctr (n, 4) and key (n, 2), got {ctr.shape} / {key.shape}")

    m0 = np.uint64(M0)
    m1 = np.uint64(M1)
    w0 = np.uint32(W0)
    w1 = np.uint32(W1)
    shift32 = np.uint64(32)

    c0 = ctr[:, 0].copy()
    c1 = ctr[:, 1].copy()
    c2 = ctr[:, 2].copy()
    c3 = ctr[:, 3].copy()
    k0 = key[:, 0].copy()
    k1 = key[:, 1].copy()
    for r in range(ROUNDS):
        if r > 0:
            k0 = k0 + w0  # uint32 + uint32 wraps
            k1 = k1 + w1
        p0 = c0.astype(np.uint64) * m0
        p1 = c2.astype(np.uint64) * m1
        hi0 = (p0 >> shift32).astype(np.uint32)
        lo0 = p0.astype(np.uint32)  # low 32 bits
        hi1 = (p1 >> shift32).astype(np.uint32)
        lo1 = p1.astype(np.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    out = np.empty((ctr.shape[0], 4), dtype=np.uint32)
    out[:, 0] = c0
    out[:, 1] = c1
    out[:, 2] = c2
    out[:, 3] = c3
    return out


def philox32_uniform_q16_np(w):
    """Vectorised philox32_uniform_q16 over a uint32 array."""
    import numpy as np

    w = np.asarray(w)
    if w.dtype != np.uint32:
        raise TypeError(f"w must be uint32, got {w.dtype}")
    return w >> np.uint32(16)


def philox32_angle_q16_np(w):
    """Vectorised philox32_angle_q16 over a uint32 array (exact, via uint64)."""
    import numpy as np

    w = np.asarray(w)
    if w.dtype != np.uint32:
        raise TypeError(f"w must be uint32, got {w.dtype}")
    return ((w.astype(np.uint64) * np.uint64(TWO_PI_Q16)) >> np.uint64(32)).astype(np.uint32)


__all__ = [
    "philox32_4x32_10",
    "philox32_uniform_q16",
    "philox32_angle_q16",
    "philox32_make_key",
    "philox32_make_counter",
    "philox32_4x32_10_np",
    "philox32_uniform_q16_np",
    "philox32_angle_q16_np",
    "TWO_PI_Q16",
    "ROUNDS",
]
