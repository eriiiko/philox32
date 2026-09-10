"""Shared fixtures: repo paths, the KAT rows, the cross-language pair derivation,
and the MSVC / nvcc toolchain lookup used to build the C++ and CUDA checkers.

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
TESTS_DIR = REPO_ROOT / "tests"
BUILD_DIR = REPO_ROOT / "build"
KAT_FILE = TESTS_DIR / "kat_philox4x32_10.txt"
XLANG_PAIRS = 64
MASK32 = 0xFFFFFFFF

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ----------------------------------------------------------------------- KAT data
def load_kat_rows(path=KAT_FILE):
    """-> list of (ctr[4], key[2], out[4]) int tuples, from the Random123 lines."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tok = line.split()
        assert tok[0] == "philox4x32" and tok[1] == "10", line
        v = [int(x, 16) for x in tok[2:]]
        assert len(v) == 10, line
        rows.append((tuple(v[0:4]), tuple(v[4:6]), tuple(v[6:10])))
    return rows


def derive_pair(rows, i):
    """The 64 cross-language inputs -- must mirror derive_pair() in tests/kat_common.h."""
    ctr_b, key_b, out_b = rows[i % len(rows)]
    ctr = tuple((out_b[j] ^ ((i * 0x9E3779B9 + j * 0x85EBCA6B) & MASK32)) for j in range(4))
    key = tuple(((key_b[j] + i * 0xC2B2AE35 + j * 0x27D4EB2F) & MASK32) for j in range(2))
    return ctr, key


@pytest.fixture(scope="session")
def kat_rows():
    rows = load_kat_rows()
    assert rows, "no philox4x32-10 vectors found in the KAT file"
    return rows


@pytest.fixture(scope="session")
def xlang_pairs(kat_rows):
    return [derive_pair(kat_rows, i) for i in range(XLANG_PAIRS)]


# ------------------------------------------------------------------ checker output
PAIR_RE = re.compile(r"^PAIR (\d+)((?: [0-9a-f]{8}){18})$")


def parse_checker_output(text):
    """-> dict(kat=(passed,total), hostdev=(agree,total)|None, pairs={i: dict}, result=str)."""
    info = {"kat": None, "hostdev": None, "pairs": {}, "result": None}
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
        else:
            m = PAIR_RE.match(line)
            if m:
                i = int(m.group(1))
                w = [int(x, 16) for x in m.group(2).split()]
                info["pairs"][i] = {
                    "ctr": tuple(w[0:4]),
                    "key": tuple(w[4:6]),
                    "out": tuple(w[6:10]),
                    "uniform": tuple(w[10:14]),
                    "angle": tuple(w[14:18]),
                }
    return info


# ---------------------------------------------------------------------- toolchain
VSWHERE = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"


def find_vcvars64():
    """-> path of vcvars64.bat, preferring VS 2022 (17.x, the toolchain nvcc 12.x
    supports), then any newer install; None when no MSVC is found."""
    if not VSWHERE.exists():
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
    """Run one command line inside the VS developer environment (vcvars64)."""
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
    """Builds tests/kat_check.cpp with MSVC (/W4 /WX, C++14) into build/; skips
    with a reason only when no compiler exists on this machine."""
    if sys.platform != "win32":
        pytest.skip("C++ KAT wrapper is written for MSVC on Windows (build it by hand elsewhere)")
    if find_vcvars64() is None:
        pytest.skip("no MSVC installation found via vswhere -- cannot build kat_check.cpp")
    BUILD_DIR.mkdir(exist_ok=True)
    exe = BUILD_DIR / "kat_check.exe"
    cmd = (
        f'cl /nologo /W4 /WX /O2 /EHsc /std:c++14 /I"{REPO_ROOT}" /I"{TESTS_DIR}" '
        f'"{TESTS_DIR / "kat_check.cpp"}" /Fe"{exe}"'  # .obj lands in cwd = build/
    )
    r = run_in_dev_prompt(cmd, BUILD_DIR)
    assert r.returncode == 0 and exe.exists(), f"MSVC build failed:\n{r.stdout}\n{r.stderr}"
    return exe


@pytest.fixture(scope="session")
def kat_check_device_exe():
    """Builds tests/kat_check_device.cu with nvcc (MSVC host compiler) into build/;
    skips when nvcc or MSVC is missing."""
    if sys.platform != "win32":
        pytest.skip("CUDA KAT wrapper is written for nvcc + MSVC on Windows")
    nvcc = find_nvcc()
    if nvcc is None:
        pytest.skip("no nvcc found (PATH, CUDA_PATH or the default toolkit dir)")
    if find_vcvars64() is None:
        pytest.skip("no MSVC installation found via vswhere -- nvcc needs a host compiler")
    BUILD_DIR.mkdir(exist_ok=True)
    exe = BUILD_DIR / "kat_check_device.exe"
    cmd = (
        f'"{nvcc}" -O2 -arch=sm_75 -std=c++14 -Xcompiler "/W4 /WX /EHsc" '
        f'-I"{REPO_ROOT}" -I"{TESTS_DIR}" "{TESTS_DIR / "kat_check_device.cu"}" -o "{exe}"'
    )
    r = run_in_dev_prompt(cmd, BUILD_DIR)
    assert r.returncode == 0 and exe.exists(), f"nvcc build failed:\n{r.stdout}\n{r.stderr}"
    return exe


def run_checker(exe):
    r = subprocess.run([str(exe), str(KAT_FILE)], capture_output=True, text=True)
    return r, parse_checker_output(r.stdout)
