"""philox32 -- Philox-4x32-10, bit-identical on CPU, CUDA device and Python.

The Python twin lives in ``philox32.philox32`` and is re-exported here; the
C++/CUDA header ships as package data next to it -- ``header_path()`` locates
it for a build system (``-I`` the parent directory, ``#include "philox32.h"``).

License: MIT, Copyright (c) 2026 Erik Steen.
"""

from pathlib import Path as _Path

from .philox32 import *  # noqa: F401,F403 -- the twin's public API
from .philox32 import __all__ as _twin_all
from .philox32 import __version__

_HEADER = _Path(__file__).resolve().parent / "philox32.h"


def header_path():
    """-> pathlib.Path of the installed philox32.h (the C++/CUDA header)."""
    if not _HEADER.exists():
        raise FileNotFoundError(f"philox32.h is not installed next to {__file__}")
    return _HEADER


def header_dir():
    """-> pathlib.Path to pass as an include directory (``-I``)."""
    return header_path().parent


__all__ = list(_twin_all) + ["header_path", "header_dir", "__version__"]
