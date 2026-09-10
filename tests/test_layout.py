"""The recommended key / counter layout -- a wire format, frozen at publication.

Properties protected: the seed is the whole key and agent_id lives in counter
word 3, so a seed x agent sweep never shares a stream (the 0.1.0-review finding:
`seed_hi ^ agent_id` collapsed seed=1/agent=2 onto seed=2/agent=1); and
philox32_draw is exactly the block function on (make_counter, make_key).
"""

import numpy as np
import pytest

import philox32 as px


def test_key_is_the_seed_and_agent_is_counter_word_3():
    assert px.philox32_make_key(0x11111111, 0x22222222) == (0x11111111, 0x22222222)
    assert px.philox32_make_counter(5, 3, 9, 7) == (5, 3, 9, 7)


@pytest.mark.parametrize("seed_word", ["seed_lo", "seed_hi"])
def test_seed_by_agent_sweep_yields_distinct_streams(seed_word):
    # Both seed words: the 0.1.0-review layout collapsed the sweep only when the
    # seed varied in the word agent_id was XORed into (seed_hi), so a sweep over
    # seed_lo alone would not have caught it.
    cells = {}
    draws = set()
    for seed in range(8):
        for agent in range(8):
            key = px.philox32_make_key(**{seed_word: seed, ("seed_hi" if seed_word == "seed_lo" else "seed_lo"): 0})
            ctr = px.philox32_make_counter(tick=1, draw_index=0, stream_salt=0, agent_id=agent)
            cells[(seed, agent)] = (key, ctr)
            draws.add(px.philox32_4x32_10(ctr, key))
    assert len(set(cells.values())) == 64, "some (seed, agent) cells share a (key, ctr)"
    assert len(draws) == 64, "some (seed, agent) cells share draws"


def test_seed_1_agent_2_differs_from_seed_2_agent_1():
    a = px.philox32_draw(seed_lo=1, seed_hi=0, agent_id=2, tick=0, draw_index=0, stream_salt=0)
    b = px.philox32_draw(seed_lo=2, seed_hi=0, agent_id=1, tick=0, draw_index=0, stream_salt=0)
    assert a != b
    # and the same for the high seed word, where the old XOR layout lived
    a = px.philox32_draw(seed_lo=0, seed_hi=1, agent_id=2, tick=0, draw_index=0, stream_salt=0)
    b = px.philox32_draw(seed_lo=0, seed_hi=2, agent_id=1, tick=0, draw_index=0, stream_salt=0)
    assert a != b


def test_every_layout_axis_changes_the_draw():
    base = dict(seed_lo=1, seed_hi=2, agent_id=3, tick=4, draw_index=5, stream_salt=6)
    ref = px.philox32_draw(**base)
    for axis in base:
        bumped = dict(base)
        bumped[axis] += 1
        assert px.philox32_draw(**bumped) != ref, axis


def test_draw_is_the_block_function_on_the_layout():
    args = dict(seed_lo=0xDEADBEEF, seed_hi=0x01234567, agent_id=42, tick=1000, draw_index=3, stream_salt=7)
    expected = px.philox32_4x32_10(
        px.philox32_make_counter(args["tick"], args["draw_index"], args["stream_salt"], args["agent_id"]),
        px.philox32_make_key(args["seed_lo"], args["seed_hi"]),
    )
    assert px.philox32_draw(**args) == expected


def test_strict_twin_accepts_numpy_ints_rejects_bool_float_and_range():
    w = np.uint32(0xFFFFFFFF)
    assert px.philox32_uniform_q16(w) == 65535
    assert px.philox32_4x32_10((np.uint32(0),) * 4, (np.uint64(0), np.int64(0))) == px.philox32_4x32_10((0,) * 4, (0, 0))
    assert isinstance(px.philox32_make_key(np.uint32(1), np.uint8(2))[0], int)
    with pytest.raises(ValueError):
        px.philox32_uniform_q16(True)
    with pytest.raises(ValueError):
        px.philox32_uniform_q16(1.0)
    with pytest.raises(ValueError):
        px.philox32_make_counter(0, 1 << 32, 0, 0)
    with pytest.raises(ValueError):
        px.philox32_make_key(-1, 0)
