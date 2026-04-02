"""
nova.crypto — Cryptography constellation (prototype)

This is *not* a full cryptography suite; it provides basic hashing/HMAC
useful for demos and tooling.
"""

from __future__ import annotations

import hashlib
import hmac


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def hmac_sha256(key: bytes, data: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()

