"""Python twin vs Random123's published known-answer vectors, scalar and numpy.

Property protected: philox32_4x32_10 IS Philox-4x32-10 as published.  Any change
to the round function, the constants, the round count or the key-bump ordering
breaks the KAT lines (which we did not produce).
"""

import numpy as np
import pytest

import philox32 as px
from conftest import MASK32, XLANG_PAIRS


def test_kat_file_has_vectors(kat_rows):
    # Random123 publishes three philox4x32-10 lines; the gate is "every line
    # present passes", not a pinned count -- but zero lines would be vacuous.
    assert len(kat_rows) >= 1


def test_scalar_matches_every_kat_line(kat_rows):
    for ctr, key, expected in kat_rows:
        assert px.philox32_4x32_10(ctr, key) == expected, (ctr, key)


def test_numpy_matches_every_kat_line(kat_rows):
    ctr = np.array([r[0] for r in kat_rows], dtype=np.uint32)
    key = np.array([r[1] for r in kat_rows], dtype=np.uint32)
    expected = np.array([r[2] for r in kat_rows], dtype=np.uint32)
    got = px.philox32_4x32_10_np(ctr, key)
    assert got.dtype == np.uint32 and got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_numpy_equals_scalar_on_xlang_pairs(xlang_pairs):
    ctr = np.array([p[0] for p in xlang_pairs], dtype=np.uint32)
    key = np.array([p[1] for p in xlang_pairs], dtype=np.uint32)
    got = px.philox32_4x32_10_np(ctr, key)
    for i, (c, k) in enumerate(xlang_pairs):
        assert tuple(int(x) for x in got[i]) == px.philox32_4x32_10(c, k), i


def test_numpy_equals_scalar_on_wraparound_heavy_inputs():
    # Inputs chosen so every intermediate hits the 32-bit wrap: all-ones, high bit
    # only, and a spread of large values -- the regime where an implicit
    # int64/float promotion would silently diverge from the scalar twin.
    words = [0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0xDEADBEEF, 0x00000001, 0xFFFFFFFE]
    pairs = []
    for a in words:
        for b in words:
            pairs.append(((a, b, a ^ b, (a + b) & MASK32), (b, a)))
    ctr = np.array([p[0] for p in pairs], dtype=np.uint32)
    key = np.array([p[1] for p in pairs], dtype=np.uint32)
    got = px.philox32_4x32_10_np(ctr, key)
    for i, (c, k) in enumerate(pairs):
        assert tuple(int(x) for x in got[i]) == px.philox32_4x32_10(c, k), (c, k)


def test_outputs_are_32_bit_and_pure(xlang_pairs):
    for c, k in xlang_pairs:
        out = px.philox32_4x32_10(c, k)
        assert len(out) == 4 and all(0 <= w <= MASK32 for w in out)
        assert px.philox32_4x32_10(c, k) == out  # stateless: same inputs, same bits


def test_xlang_pairs_are_distinct(xlang_pairs):
    assert len(xlang_pairs) == XLANG_PAIRS
    assert len(set(xlang_pairs)) == XLANG_PAIRS


def test_scalar_rejects_out_of_range_inputs():
    with pytest.raises(ValueError):
        px.philox32_4x32_10((1 << 32, 0, 0, 0), (0, 0))
    with pytest.raises(ValueError):
        px.philox32_4x32_10((0, 0, 0, 0), (-1, 0))
    with pytest.raises(ValueError):
        px.philox32_4x32_10((0, 0, 0), (0, 0))
    with pytest.raises(ValueError):
        px.philox32_4x32_10((0.5, 0, 0, 0), (0, 0))


def test_numpy_rejects_non_uint32():
    with pytest.raises(TypeError):
        px.philox32_4x32_10_np(np.zeros((1, 4), dtype=np.int64), np.zeros((1, 2), dtype=np.uint32))
    with pytest.raises(TypeError):
        px.philox32_4x32_10_np(np.zeros((1, 4), dtype=np.uint32), np.zeros((1, 2), dtype=np.float64))
    with pytest.raises(ValueError):
        px.philox32_4x32_10_np(np.zeros((2, 4), dtype=np.uint32), np.zeros((1, 2), dtype=np.uint32))


def test_recommended_layout_helpers():
    assert px.philox32_make_key(0x11111111, 0x22222222, 0x00000007) == (0x11111111, 0x22222225)
    assert px.philox32_make_counter(5, 3, 9) == (5, 3, 9, 0)
    # Different agents / salts / draws give different blocks (independent streams).
    base = px.philox32_4x32_10(px.philox32_make_counter(1, 0, 0), px.philox32_make_key(1, 2, 0))
    assert px.philox32_4x32_10(px.philox32_make_counter(1, 0, 0), px.philox32_make_key(1, 2, 1)) != base
    assert px.philox32_4x32_10(px.philox32_make_counter(1, 1, 0), px.philox32_make_key(1, 2, 0)) != base
    assert px.philox32_4x32_10(px.philox32_make_counter(1, 0, 1), px.philox32_make_key(1, 2, 0)) != base
