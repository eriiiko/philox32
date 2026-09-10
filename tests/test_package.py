"""Packaging invariants: the version is kept in one place per artifact and they
agree, and the installed package carries the C++ header.
"""

import re
from pathlib import Path

import philox32 as px
from conftest import PKG_DIR, REPO_ROOT


def _pyproject_version():
    text = (REPO_ROOT / "pyproject.toml").read_text()
    try:
        import tomllib  # 3.11+
        return tomllib.loads(text)["project"]["version"]
    except ImportError:
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        assert m, "no version in pyproject.toml"
        return m.group(1)


def test_version_matches_pyproject():
    assert px.__version__ == _pyproject_version()


def test_header_path_points_at_the_shipped_header():
    p = px.header_path()
    assert isinstance(p, Path) and p.is_file() and p.name == "philox32.h"
    assert p == PKG_DIR / "philox32.h"
    assert px.header_dir() == p.parent
    text = p.read_text()
    assert "PHILOX32_H" in text and "philox32_4x32_10" in text


def test_public_api_is_reexported():
    for name in ("philox32_4x32_10", "philox32_draw", "philox32_below", "philox32_uniform_q16",
                 "philox32_angle_q16", "philox32_4x32_10_np", "TWO_PI_Q16", "header_path"):
        assert name in px.__all__ and hasattr(px, name), name
