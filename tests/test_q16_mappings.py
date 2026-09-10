"""The two Q16.16 output mappings: ranges, exact endpoints, monotonicity, and
scalar/numpy agreement.

Property protected: uniform_q16 covers exactly [0, 65536) and angle_q16 exactly
[0, TWO_PI_Q16), both as pure integer maps of the 32-bit word.
"""

import math

import numpy as np
import pytest

import philox32 as px

MAX32 = 0xFFFFFFFF
EDGE_WORDS = [0, 1, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, 0xFFFF0000, 0xFFFEFFFF, MAX32 - 1, MAX32]


def test_two_pi_constant_is_round_of_2pi_times_65536():
    # Documented derivation of the checked-in constant (a test-time float is fine;
    # the library itself never touches one).
    assert px.TWO_PI_Q16 == round(2 * math.pi * 65536) == 411775


def test_uniform_exact_endpoints():
    assert px.philox32_uniform_q16(0) == 0
    assert px.philox32_uniform_q16(0xFFFF) == 0
    assert px.philox32_uniform_q16(0x10000) == 1
    assert px.philox32_uniform_q16(0x80000000) == 32768  # 0.5
    assert px.philox32_uniform_q16(MAX32) == 65535


def test_angle_exact_endpoints():
    assert px.philox32_angle_q16(0) == 0
    assert px.philox32_angle_q16(0x80000000) == px.TWO_PI_Q16 // 2  # pi
    assert px.philox32_angle_q16(MAX32) == px.TWO_PI_Q16 - 1
    assert px.philox32_angle_q16(MAX32) < px.TWO_PI_Q16


@pytest.mark.parametrize("w", EDGE_WORDS)
def test_ranges_at_edges(w):
    assert 0 <= px.philox32_uniform_q16(w) < 65536
    assert 0 <= px.philox32_angle_q16(w) < px.TWO_PI_Q16


def test_ranges_and_monotonicity_over_a_sweep():
    # A dense uint32 sweep (every 2^12-th word plus the top) through the numpy
    # variant, and the scalar twin on the same words.
    w = np.arange(0, 1 << 32, 1 << 12, dtype=np.uint64).astype(np.uint32)
    w = np.append(w, np.uint32(MAX32))
    u = px.philox32_uniform_q16_np(w)
    a = px.philox32_angle_q16_np(w)
    assert u.dtype == np.uint32 and a.dtype == np.uint32
    assert u.min() == 0 and u.max() == 65535
    assert a.min() == 0 and a.max() == px.TWO_PI_Q16 - 1
    assert np.all(np.diff(u.astype(np.int64)) >= 0)
    assert np.all(np.diff(a.astype(np.int64)) >= 0)
    for i in range(0, len(w), 97):
        wi = int(w[i])
        assert px.philox32_uniform_q16(wi) == int(u[i])
        assert px.philox32_angle_q16(wi) == int(a[i])


def test_mappings_on_philox_output(xlang_pairs):
    for c, k in xlang_pairs:
        for word in px.philox32_4x32_10(c, k):
            assert 0 <= px.philox32_uniform_q16(word) < 65536
            assert 0 <= px.philox32_angle_q16(word) < px.TWO_PI_Q16


def test_mappings_reject_out_of_range():
    with pytest.raises(ValueError):
        px.philox32_uniform_q16(1 << 32)
    with pytest.raises(ValueError):
        px.philox32_angle_q16(-1)
    with pytest.raises(TypeError):
        px.philox32_angle_q16_np(np.array([1.0]))
