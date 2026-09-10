"""philox32_below(w, n): integer in [0, n) by Lemire's multiply-shift.

Properties protected: below IS floor(w * n / 2**32) exactly (bin boundaries
land where the formula says), its range is [0, n) for every n including 0, 1
and 2**32-1, it is monotone in w, and scalar and numpy agree.
"""

import numpy as np
import pytest

import philox32 as px

MAX32 = 0xFFFFFFFF
NS = [0, 1, 2, 3, 6, 7, 10, 100, 1000, 65536, 1_000_000, 0x7FFFFFFF, 0x80000000, MAX32]
WORDS = [0, 1, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFF0000, MAX32 - 1, MAX32]


@pytest.mark.parametrize("n", NS)
def test_range_for_every_n(n):
    for w in WORDS:
        b = px.philox32_below(w, n)
        assert 0 <= b < max(n, 1), (w, n)
    if n:
        assert px.philox32_below(MAX32, n) == n - 1
        assert px.philox32_below(0, n) == 0


@pytest.mark.parametrize("n", [n for n in NS if n >= 2])
def test_bin_boundaries_are_exactly_floor_of_w_n_over_2_32(n):
    # Value k first appears at w = ceil(k * 2**32 / n); the word before maps to k-1.
    for k in sorted({1, 2, n // 2, n - 1} - {0}):
        w_first = -(-(k << 32) // n)
        if w_first > MAX32:
            continue
        assert px.philox32_below(w_first, n) == k, (n, k)
        assert px.philox32_below(w_first - 1, n) == k - 1, (n, k)


def test_monotone_in_w_and_numpy_agrees():
    w = np.arange(0, 1 << 32, 1 << 12, dtype=np.uint64).astype(np.uint32)
    w = np.append(w, np.uint32(MAX32))
    for n in (6, 1000, MAX32):
        b = px.philox32_below_np(w, np.uint32(n))
        assert b.dtype == np.uint32
        assert b.min() == 0 and b.max() == n - 1
        assert np.all(np.diff(b.astype(np.int64)) >= 0)
        for i in range(0, len(w), 89):
            assert px.philox32_below(int(w[i]), n) == int(b[i])
    # n as an array, broadcast elementwise
    ns = np.array([1, 6, 1000, MAX32], dtype=np.uint32)
    ws = np.array([MAX32, MAX32, 0, MAX32], dtype=np.uint32)
    assert px.philox32_below_np(ws, ns).tolist() == [0, 5, 0, MAX32 - 1]


def test_bias_bound_for_small_n():
    # For n = 6 the bins hold floor or ceil of 2**32/6 words -- verify the bin
    # sizes from the boundary formula rather than by enumerating 2**32 words.
    # The documented bound is the fattest bin's excess over the IDEAL bin 2**32/n:
    # (max - 2**32/n) / (2**32/n) < n / 2**32, i.e. in integers max*n - 2**32 < n.
    for n in (6, 7, 1000, 1_000_000):
        edges = [-(-(k << 32) // n) for k in range(n)] + [1 << 32]
        sizes = [edges[k + 1] - edges[k] for k in range(n)]
        assert max(sizes) - min(sizes) <= 1, n
        assert 0 <= max(sizes) * n - (1 << 32) < n, n


def test_below_rejects_out_of_range():
    with pytest.raises(ValueError):
        px.philox32_below(1 << 32, 5)
    with pytest.raises(ValueError):
        px.philox32_below(5, -1)
    with pytest.raises(TypeError):
        px.philox32_below_np(np.array([1], dtype=np.uint32), np.int64(5))
