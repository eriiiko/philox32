"""The C++ header, built with the platform compiler and run against the KAT
file, plus the bit-for-bit cross-language comparison with the Python twin on
64 pairs and 8 layout draws.

Properties protected: philox32.h IS Philox-4x32-10 (KAT), and philox32.h and
philox32.py are the same function (cross-language identity) -- block function,
the three output mappings, and the key/counter layout composition.
"""

import pytest

import philox32 as px
from conftest import XLANG_DRAWS, XLANG_PAIRS, run_checker


@pytest.fixture(scope="module")
def cpp_run(kat_check_exe):
    r, info = run_checker(kat_check_exe)
    assert info["result"] is not None, f"no RESULT line:\n{r.stdout}\n{r.stderr}"
    return r, info


def test_cpp_kat_passes(cpp_run, kat_rows):
    r, info = cpp_run
    assert info["kat"] == (len(kat_rows), len(kat_rows)), r.stdout
    assert info["result"] == "PASS", r.stdout
    assert r.returncode == 0


def test_cpp_prints_the_agreed_pairs(cpp_run, xlang_pairs):
    _, info = cpp_run
    assert sorted(info["pairs"]) == list(range(XLANG_PAIRS))
    for i, (ctr, key) in enumerate(xlang_pairs):
        assert info["pairs"][i]["ctr"] == ctr, i
        assert info["pairs"][i]["key"] == key, i


def test_cpp_and_python_agree_bit_for_bit(cpp_run):
    _, info = cpp_run
    for i, p in info["pairs"].items():
        assert px.philox32_4x32_10(p["ctr"], p["key"]) == p["out"], i


def test_cpp_and_python_mappings_agree(cpp_run):
    _, info = cpp_run
    for i, p in info["pairs"].items():
        out = p["out"]
        assert tuple(px.philox32_uniform_q16(w) for w in out) == p["uniform"], i
        assert tuple(px.philox32_angle_q16(w) for w in out) == p["angle"], i
        assert tuple(px.philox32_below(out[j], out[(j + 1) % 4]) for j in range(4)) == p["below"], i
        assert all(a < px.TWO_PI_Q16 for a in p["angle"])
        assert all(u < 65536 for u in p["uniform"])


def test_cpp_and_python_layout_draws_agree(cpp_run, xlang_draws):
    _, info = cpp_run
    assert sorted(info["draws"]) == list(range(XLANG_DRAWS))
    for i, inputs in enumerate(xlang_draws):
        d = info["draws"][i]
        assert d["in"] == inputs, i
        seed_lo, seed_hi, agent_id, tick, draw_index, salt = inputs
        assert px.philox32_draw(seed_lo, seed_hi, agent_id, tick, draw_index, salt) == d["out"], i
