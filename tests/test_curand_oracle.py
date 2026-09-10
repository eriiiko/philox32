"""philox32.h vs NVIDIA cuRAND's curand_Philox4x32_10 -- an independent
implementation of the same primitive -- on 200 000 fuzzed pairs plus 12
adversarial edge rows, and the Python twin on the cuRAND edge outputs.

This is the answer to "three published vectors is thin": the oracle shares no
arithmetic with the code under test.  Skips when nvcc / a GPU is absent.
"""

import subprocess

import pytest

import philox32 as px
from conftest import parse_checker_output


@pytest.fixture(scope="module")
def oracle_run(curand_oracle_exe):
    r = subprocess.run([str(curand_oracle_exe)], capture_output=True, text=True)
    info = parse_checker_output(r.stdout)
    if info["result"] == "NO_DEVICE":
        pytest.skip("curand_oracle built, but there is no CUDA device on this machine")
    assert info["result"] is not None, f"no RESULT line:\n{r.stdout}\n{r.stderr}"
    return r, info


def test_curand_agrees_on_every_fuzzed_pair(oracle_run):
    r, info = oracle_run
    assert info["oracle"] is not None, r.stdout
    assert info["oracle"]["n"] >= 100_000
    assert info["oracle"]["edge"] >= 12
    assert info["oracle"]["mismatches"] == 0, r.stdout
    assert info["oracle"]["edge_mismatches"] == 0, r.stdout
    assert info["result"] == "PASS" and r.returncode == 0


def test_python_twin_reproduces_curand_edge_rows(oracle_run):
    _, info = oracle_run
    assert len(info["edges"]) == info["oracle"]["edge"]
    for e in info["edges"]:
        assert px.philox32_4x32_10(e["ctr"], e["key"]) == e["out"], e
