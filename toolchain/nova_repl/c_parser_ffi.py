"""
ctypes bridge to the C parser shared library (Option 2).

This module is intentionally small: load the DLL/.so and expose `parse_dump`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Tuple


class CParserFFIError(RuntimeError):
    pass


def _default_library_path() -> Path:
    """
    CMake default output (Windows):
      compiler/build/parser/Debug/nova_parser_ffi.dll
    or:
      compiler/build/parser/Release/nova_parser_ffi.dll
    """
    root = Path(__file__).resolve().parents[2]
    build = root / "compiler" / "build" / "parser"
    # try common configs
    for cfg in ("Debug", "Release", "RelWithDebInfo", "MinSizeRel"):
        p = build / cfg / "nova_parser_ffi.dll"
        if p.exists():
            return p
    # fallback: search a couple of likely spots
    p = build / "nova_parser_ffi.dll"
    return p


def _load() -> ctypes.CDLL:
    env = os.environ.get("NOVA_PARSER_FFI_LIB")
    lib_path = Path(env) if env else _default_library_path()
    if not lib_path.exists():
        raise CParserFFIError(
            "C parser FFI library not found. Build it with CMake under `compiler/`.\n"
            f"Tried: {lib_path}\n"
            "Or set environment variable NOVA_PARSER_FFI_LIB to the dll path."
        )
    lib = ctypes.CDLL(str(lib_path))

    lib.nova_parse_dump.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.nova_parse_dump.restype = ctypes.c_int

    lib.nova_ffi_free.argtypes = [ctypes.c_char_p]
    lib.nova_ffi_free.restype = None
    return lib


def parse_dump(src: str, filename: str = "<repl>") -> Tuple[int, str, str]:
    """
    Returns (status, dump, errors).

    status:
      0 = success
      1 = parsed but had lexer/parser errors
      2 = internal/argument error
    """
    lib = _load()
    src_b = src.encode("utf-8")
    fn_b = filename.encode("utf-8")

    out_dump = ctypes.c_char_p()
    out_err = ctypes.c_char_p()
    status = int(
        lib.nova_parse_dump(
            ctypes.c_char_p(src_b),
            ctypes.c_size_t(len(src_b)),
            ctypes.c_char_p(fn_b),
            ctypes.byref(out_dump),
            ctypes.byref(out_err),
        )
    )

    try:
        dump = out_dump.value.decode("utf-8", errors="replace") if out_dump.value else ""
        err = out_err.value.decode("utf-8", errors="replace") if out_err.value else ""
        return status, dump, err
    finally:
        if out_dump.value:
            lib.nova_ffi_free(out_dump)
        if out_err.value:
            lib.nova_ffi_free(out_err)

