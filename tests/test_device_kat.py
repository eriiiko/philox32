"""The same header compiled as CUDA device code (nvcc), run on the GPU: KAT,
host-vs-device identity inside one binary, and cross-language identity with the
Python twin (block function, mappings, layout draws).  Skips, with the reason,
when nvcc or MSVC is missing or there is no CUDA device; any other CUDA error
is a failure, not a skip.
"""

import pytest

import philox32 as px
from conftest import XLANG_DRAWS, XLANG_PAIRS, run_checker


@pytest.fixture(scope="module")
def device_run(kat_check_device_exe):
    r, info = run_checker(kat_check_device_exe)
    if info["result"] == "NO_DEVICE":
        pytest.skip("kat_check_device built, but there is no CUDA device on this machine")
    assert info["result"] is not None, f"no RESULT line:\n{r.stdout}\n{r.stderr}"
    return r, info


def test_device_kat_passes(device_run, kat_rows):
    r, info = device_run
    assert info["kat"] == (len(kat_rows), len(kat_rows)), r.stdout
    assert info["result"] == "PASS", r.stdout + r.stderr
    assert r.returncode == 0


def test_device_equals_host_in_same_binary(device_run, kat_rows):
    _, info = device_run
    total = len(kat_rows) + XLANG_PAIRS
    assert info["hostdev"] == (total, total)


def test_device_and_python_agree_bit_for_bit(device_run, xlang_pairs):
    _, info = device_run
    assert sorted(info["pairs"]) == list(range(XLANG_PAIRS))
    for i, (ctr, key) in enumerate(xlang_pairs):
        p = info["pairs"][i]
        assert (p["ctr"], p["key"]) == (ctr, key), i
        out = p["out"]
        assert px.philox32_4x32_10(ctr, key) == out, i
        assert tuple(px.philox32_uniform_q16(w) for w in out) == p["uniform"], i
        assert tuple(px.philox32_angle_q16(w) for w in out) == p["angle"], i
        assert tuple(px.philox32_below(out[j], out[(j + 1) % 4]) for j in range(4)) == p["below"], i


def test_device_and_python_layout_draws_agree(device_run, xlang_draws):
    _, info = device_run
    assert sorted(info["draws"]) == list(range(XLANG_DRAWS))
    for i, inputs in enumerate(xlang_draws):
        d = info["draws"][i]
        assert d["in"] == inputs, i
        assert px.philox32_draw(*inputs) == d["out"], i
