"""Shared fixtures: repo paths, the KAT rows, the cross-language derivations,
and the compiler lookup used to build the C++ / CUDA checkers.

Conventions: tests assert properties (KAT identity, cross-language identity,
mapping ranges) -- never snapshots of this package's own output.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "philox32"          # holds philox32.h + the Python twin
TESTS_DIR = REPO_ROOT / "tests"
BUILD_DIR = REPO_ROOT / "build"
KAT_FILE = TESTS_DIR / "kat_philox4x32_10.txt"
XLANG_PAIRS = 64
XLANG_DRAWS = 8
MASK32 = 0xFFFFFFFF

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ----------------------------------------------------------------------- KAT data
def load_kat_rows(path=KAT_FILE):
    """-> list of (ctr[4], key[2], out[4]) int tuples from the `philox4x32 10` lines.

    Other ciphers' lines (the full Random123 kat_vectors file carries philox4x32 7,
    philox4x64, threefry, ars...) are filtered, mirroring load_kat() in
    kat_common.h.  Malformed matching lines raise -- never assert, so `python -O`
    cannot make the loader vacuous.
    """
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tok = line.split()
        if len(tok) < 2 or tok[0] != "philox4x32" or tok[1] != "10":
            continue
        v = [int(x, 16) for x in tok[2:]]
        if len(v) != 10 or any(not 0 <= x <= MASK32 for x in v):
            raise ValueError(f"malformed philox4x32-10 KAT line: {line!r}")
        rows.append((tuple(v[0:4]), tuple(v[4:6]), tuple(v[6:10])))
    return rows


def derive_pair(rows, i):
    """The 64 cross-language inputs -- hand-mirrors derive_pair() in tests/kat_common.h.
    Self-gating: the checker prints its ctr/key and pytest compares them to this."""
    ctr_b, key_b, out_b = rows[i % len(rows)]
    ctr = tuple((out_b[j] ^ ((i * 0x9E3779B9 + j * 0x85EBCA6B) & MASK32)) for j in range(4))
    key = tuple(((key_b[j] + i * 0xC2B2AE35 + j * 0x27D4EB2F) & MASK32) for j in range(2))
    return ctr, key


def derive_draw_inputs(i):
    """The 8 DRAW inputs (seed_lo, seed_hi, agent_id, tick, draw_index, stream_salt)
    -- hand-mirrors derive_draw_inputs() in tests/kat_common.h, self-gating as above."""
    flip = 0 if i < 4 else MASK32
    return tuple(((0x9E3779B9 * (i + 1) + 0x85EBCA6B * j) & MASK32) ^ flip for j in range(6))


@pytest.fixture(scope="session")
def kat_rows():
    rows = load_kat_rows()
    if not rows:
        raise ValueError("no philox4x32-10 vectors found in the KAT file (vacuous gate)")
    return rows


@pytest.fixture(scope="session")
def xlang_pairs(kat_rows):
    return [derive_pair(kat_rows, i) for i in range(XLANG_PAIRS)]


@pytest.fixture(scope="session")
def xlang_draws():
    return [derive_draw_inputs(i) for i in range(XLANG_DRAWS)]


# ------------------------------------------------------------------ checker output
PAIR_RE = re.compile(r"^PAIR (\d+)((?: [0-9a-f]{8}){22})$")
DRAW_RE = re.compile(r"^DRAW (\d+)((?: [0-9a-f]{8}){10})$")
EDGE_RE = re.compile(r"^EDGE((?: [0-9a-f]{8}){10})$")
ORACLE_RE = re.compile(r"^ORACLE N=(\d+) edge=(\d+) mismatches=(\d+) edge_mismatches=(\d+)$")


def parse_checker_output(text):
    """-> dict with kat, hostdev, pairs, draws, oracle, edges, result."""
    info = {"kat": None, "hostdev": None, "pairs": {}, "draws": {}, "oracle": None, "edges": [], "result": None}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("KAT "):
            a, b = line[4:].split("/")
            info["kat"] = (int(a), int(b))
        elif line.startswith("HOSTDEV "):
            a, b = line[8:].split("/")
            info["hostdev"] = (int(a), int(b))
        elif line.startswith("RESULT: "):
            info["result"] = line[8:]
        elif ORACLE_RE.match(line):
            n, e, m, me = (int(x) for x in ORACLE_RE.match(line).groups())
            info["oracle"] = {"n": n, "edge": e, "mismatches": m, "edge_mismatches": me}
        else:
            m = PAIR_RE.match(line)
            if m:
                w = [int(x, 16) for x in m.group(2).split()]
                info["pairs"][int(m.group(1))] = {
                    "ctr": tuple(w[0:4]), "key": tuple(w[4:6]), "out": tuple(w[6:10]),
                    "uniform": tuple(w[10:14]), "angle": tuple(w[14:18]), "below": tuple(w[18:22]),
                }
                continue
            m = DRAW_RE.match(line)
            if m:
                w = [int(x, 16) for x in m.group(2).split()]
                info["draws"][int(m.group(1))] = {"in": tuple(w[0:6]), "out": tuple(w[6:10])}
                continue
            m = EDGE_RE.match(line)
            if m:
                w = [int(x, 16) for x in m.group(1).split()]
                info["edges"].append({"ctr": tuple(w[0:4]), "key": tuple(w[4:6]), "out": tuple(w[6:10])})
    return info


# ---------------------------------------------------------------------- toolchain
VSWHERE = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
WORKER = os.environ.get("PYTEST_XDIST_WORKER", "")   # per-worker artifact names under pytest -n


def find_vcvars64():
    """-> path of vcvars64.bat, preferring VS 2022 (17.x, the toolchain nvcc 12.x
    supports), then any newer install; None when no MSVC is found."""
    if sys.platform != "win32" or not VSWHERE.exists():
        return None
    for version_range in ("[17.0,18.0)", "[17.0,)"):
        try:
            out = subprocess.run(
                [str(VSWHERE), "-products", "*", "-version", version_range,
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath", "-latest"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
        except OSError:
            return None
        if out:
            bat = Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if bat.exists():
                return bat
    return None


def find_unix_cxx():
    """-> a C++ compiler on non-Windows: $CXX, then c++ / g++ / clang++ on PATH."""
    env = os.environ.get("CXX")
    if env and shutil.which(env):
        return env
    for cand in ("c++", "g++", "clang++"):
        if shutil.which(cand):
            return cand
    return None


def find_nvcc():
    exe = shutil.which("nvcc")
    if exe:
        return Path(exe)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and (Path(cuda_path) / "bin" / "nvcc.exe").exists():
        return Path(cuda_path) / "bin" / "nvcc.exe"
    root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if root.exists():
        for d in sorted(root.glob("v*"), reverse=True):
            if (d / "bin" / "nvcc.exe").exists():
                return d / "bin" / "nvcc.exe"
    return None


def run_in_dev_prompt(command, cwd):
    """Run one command line inside the VS developer environment (vcvars64) on
    Windows, or through the shell elsewhere."""
    if sys.platform != "win32":
        return subprocess.run(command, cwd=str(cwd), shell=True, capture_output=True, text=True)
    vcvars = find_vcvars64()
    if vcvars is None:
        raise RuntimeError("no MSVC found")
    # One verbatim string: a list would make subprocess re-escape the inner
    # quotes, which cmd.exe then misparses.  /s strips the outer quote pair.
    full = f'cmd.exe /s /c ""{vcvars}" >nul 2>&1 && {command}"'
    return subprocess.run(full, cwd=str(cwd), capture_output=True, text=True)


@pytest.fixture(scope="session")
def kat_check_exe():
    """Builds tests/kat_check.cpp: MSVC (/W4 /WX, C++14 -- MSVC has no C++11 mode)
    on Windows, else $CXX / c++ / g++ / clang++ at the advertised C++11 floor with
    -Wall -Wextra -Werror -pedantic.  Skips, with the reason, only when no compiler
    exists at all."""
    BUILD_DIR.mkdir(exist_ok=True)
    src = TESTS_DIR / "kat_check.cpp"
    if sys.platform == "win32":
        if find_vcvars64() is None:
            pytest.skip("no MSVC installation found via vswhere -- cannot build kat_check.cpp")
        exe = BUILD_DIR / f"kat_check{WORKER}.exe"
        cmd = (f'cl /nologo /W4 /WX /O2 /EHsc /std:c++14 /I"{PKG_DIR}" /I"{TESTS_DIR}" '
               f'"{src}" /Fo"{BUILD_DIR / ("kat_check" + WORKER + ".obj")}" /Fe"{exe}"')
    else:
        cxx = find_unix_cxx()
        if cxx is None:
            pytest.skip("no C++ compiler found ($CXX, c++, g++, clang++) -- cannot build kat_check.cpp")
        exe = BUILD_DIR / f"kat_check{WORKER}"
        cmd = (f'{cxx} -std=c++11 -O2 -Wall -Wextra -Werror -pedantic -I"{PKG_DIR}" -I"{TESTS_DIR}" '
               f'"{src}" -o "{exe}"')
    r = run_in_dev_prompt(cmd, BUILD_DIR)
    assert r.returncode == 0 and exe.exists(), f"C++ build failed:\n{cmd}\n{r.stdout}\n{r.stderr}"
    return exe


def _build_cuda(src_name, exe_stem):
    """nvcc build of tests/<src_name> into build/; -arch=native first (the GPU that
    is actually here), falling back to sm_75 (PTX-forward-compatible) if that
    fails, e.g. on a build box with no GPU."""
    nvcc = find_nvcc()
    if nvcc is None:
        pytest.skip("no nvcc found (PATH, CUDA_PATH or the default toolkit dir)")
    if sys.platform == "win32" and find_vcvars64() is None:
        pytest.skip("no MSVC installation found via vswhere -- nvcc needs a host compiler")
    BUILD_DIR.mkdir(exist_ok=True)
    exe = BUILD_DIR / (exe_stem + WORKER + (".exe" if sys.platform == "win32" else ""))
    host_flags = '-Xcompiler "/W4 /WX /EHsc"' if sys.platform == "win32" else '-Xcompiler "-Wall -Wextra"'
    logs = []
    for arch in ("native", "sm_75"):
        cmd = (f'"{nvcc}" -O2 -arch={arch} -std=c++14 {host_flags} -I"{PKG_DIR}" -I"{TESTS_DIR}" '
               f'"{TESTS_DIR / src_name}" -o "{exe}"')
        r = run_in_dev_prompt(cmd, BUILD_DIR)
        logs.append(f"{cmd}\n{r.stdout}\n{r.stderr}")
        if r.returncode == 0 and exe.exists():
            return exe
    raise AssertionError("nvcc build failed:\n" + "\n---\n".join(logs))


@pytest.fixture(scope="session")
def kat_check_device_exe():
    return _build_cuda("kat_check_device.cu", "kat_check_device")


@pytest.fixture(scope="session")
def curand_oracle_exe():
    return _build_cuda("curand_oracle.cu", "curand_oracle")


def run_checker(exe):
    r = subprocess.run([str(exe), str(KAT_FILE)], capture_output=True, text=True)
    return r, parse_checker_output(r.stdout)
