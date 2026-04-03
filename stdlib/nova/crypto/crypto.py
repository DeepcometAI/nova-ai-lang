"""
nova.crypto — Cryptography constellation (Python prototype)

Basic hashing, HMAC, UUID generation, and base64 encoding.
This is NOT a full cryptography suite — it wraps Python stdlib only
and is suitable for demos, tooling, and test fixtures.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as _hmac
import os
import secrets
import uuid
from typing import Optional, Union

__all__ = [
    "sha256",
    "sha512",
    "sha1",
    "md5",
    "hmac_sha256",
    "hmac_sha512",
    "random_bytes",
    "random_hex",
    "random_token",
    "uuid4",
    "base64_encode",
    "base64_decode",
    "compare_digest",
    "pbkdf2",
]


def sha256(data: Union[bytes, str]) -> str:
    """SHA-256 hex digest of bytes or UTF-8 string."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha512(data: Union[bytes, str]) -> str:
    """SHA-512 hex digest."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha512(data).hexdigest()


def sha1(data: Union[bytes, str]) -> str:
    """SHA-1 hex digest (not collision-resistant; avoid for security)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def md5(data: Union[bytes, str]) -> str:
    """MD5 hex digest (not collision-resistant; avoid for security)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest()


def hmac_sha256(key: Union[bytes, str], data: Union[bytes, str]) -> str:
    """HMAC-SHA256 hex digest."""
    if isinstance(key, str):
        key = key.encode("utf-8")
    if isinstance(data, str):
        data = data.encode("utf-8")
    return _hmac.new(key, data, hashlib.sha256).hexdigest()


def hmac_sha512(key: Union[bytes, str], data: Union[bytes, str]) -> str:
    """HMAC-SHA512 hex digest."""
    if isinstance(key, str):
        key = key.encode("utf-8")
    if isinstance(data, str):
        data = data.encode("utf-8")
    return _hmac.new(key, data, hashlib.sha512).hexdigest()


def random_bytes(n: int) -> bytes:
    """Cryptographically random n bytes."""
    return secrets.token_bytes(int(n))


def random_hex(n_bytes: int = 16) -> str:
    """Cryptographically random hex string (2*n_bytes chars)."""
    return secrets.token_hex(int(n_bytes))


def random_token(n_bytes: int = 32) -> str:
    """URL-safe base64 token from n_bytes of random data."""
    return secrets.token_urlsafe(int(n_bytes))


def uuid4() -> str:
    """Generate a random UUID4 string."""
    return str(uuid.uuid4())


def base64_encode(data: Union[bytes, str]) -> str:
    """Standard base64 encode to ASCII string."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return base64.b64encode(data).decode("ascii")


def base64_decode(data: Union[bytes, str]) -> bytes:
    """Standard base64 decode to bytes."""
    if isinstance(data, str):
        data = data.encode("ascii")
    return base64.b64decode(data)


def compare_digest(a: Union[bytes, str], b: Union[bytes, str]) -> bool:
    """Constant-time comparison (prevents timing attacks)."""
    if isinstance(a, str) and isinstance(b, str):
        return _hmac.compare_digest(a, b)
    if isinstance(a, str):
        a = a.encode("utf-8")
    if isinstance(b, str):
        b = b.encode("utf-8")
    return _hmac.compare_digest(a, b)


def pbkdf2(
    password: Union[bytes, str],
    salt: Union[bytes, str],
    iterations: int = 200_000,
    dk_len: int = 32,
    hash_name: str = "sha256",
) -> str:
    """
    PBKDF2 key derivation, returns hex-encoded derived key.
    """
    if isinstance(password, str):
        password = password.encode("utf-8")
    if isinstance(salt, str):
        salt = salt.encode("utf-8")
    dk = hashlib.pbkdf2_hmac(
        hash_name, password, salt, int(iterations), dklen=int(dk_len)
    )
    return dk.hex()
