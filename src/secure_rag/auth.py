"""Signed bearer tokens: who the caller is comes from the server, never from the request body.

A token is ``base64url(claims).base64url(HMAC-SHA256(secret, base64url(claims)))``. The claims
name the user, the tenant, the roles and an expiry. Group membership is deliberately *not* in
the token: it is looked up in the directory on every request, so a team move takes effect
without waiting for tokens to expire. In production the verifier would check an IdP's JWTs; the
shape of the check is the same.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

READER, WRITER, AUDITOR = "reader", "writer", "auditor"


class TokenError(ValueError):
    """The token is missing, malformed, forged or expired."""


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    roles: frozenset[str]


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TokenSigner:
    def __init__(self, secret: bytes | str, clock: Callable[[], float] = time.time) -> None:
        secret = secret.encode() if isinstance(secret, str) else secret
        if len(secret) < 16:
            raise ValueError("the token secret must be at least 16 bytes")
        self._secret = secret
        self._clock = clock

    def _sign(self, payload: str) -> str:
        return _b64(hmac.new(self._secret, payload.encode(), hashlib.sha256).digest())

    def issue(self, user_id: str, tenant_id: str, roles: Iterable[str] = (READER,), ttl_seconds: int = 3600) -> str:
        claims = {
            "sub": user_id,
            "tenant": tenant_id,
            "roles": sorted(set(roles)),
            "exp": int(self._clock()) + ttl_seconds,
        }
        payload = _b64(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
        return f"{payload}.{self._sign(payload)}"

    def verify(self, token: str) -> Principal:
        try:
            payload, signature = token.split(".")
        except ValueError as error:
            raise TokenError("malformed token") from error
        if not hmac.compare_digest(signature, self._sign(payload)):
            raise TokenError("bad signature")
        try:
            claims = json.loads(_unb64(payload))
            principal = Principal(str(claims["sub"]), str(claims["tenant"]), frozenset(map(str, claims["roles"])))
            expires = float(claims["exp"])
        except (ValueError, KeyError, TypeError) as error:
            raise TokenError("malformed claims") from error
        if expires <= self._clock():
            raise TokenError("token expired")
        if not principal.user_id or not principal.tenant_id:
            raise TokenError("token names no user or tenant")
        return principal
