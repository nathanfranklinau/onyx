"""Unit tests for MCP OAuth token-expiry persistence and provider hydration.

These tests cover the fix for the "stored tokens look valid forever"
class of bugs:

1. The SDK's `OAuthToken.expires_in` is the lifetime at issuance, not
   remaining lifetime. After persisting and reloading hours later, the
   value is meaningless without an absolute expiry stamp.
2. `OAuthClientProvider._initialize` loads tokens from storage but
   never calls `update_token_expiry`, so a freshly-constructed
   provider treats stored tokens as valid indefinitely and skips the
   refresh branch — every 401 cascades into a full re-auth.

The fix:

- Persist an absolute expiry timestamp alongside the token (derived
  from `expires_in`, falling back to the JWT `exp` claim for IdPs like
  Salesforce that omit `expires_in`).
- Recompute `expires_in` as `abs_expiry - now` on read, so the SDK's
  `update_token_expiry` sees a meaningful relative remaining lifetime.
- Subclass `OAuthClientProvider` to seed `token_expiry_time` after
  storage hydration so `is_token_valid()` returns False once the
  stamped token has actually expired.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata
from mcp.shared.auth import OAuthClientInformationFull
from mcp.shared.auth import OAuthToken
from pydantic import AnyUrl

from onyx.server.features.mcp.api import _ABS_EXPIRY_KEY
from onyx.server.features.mcp.api import _attach_absolute_expiry
from onyx.server.features.mcp.api import _compute_absolute_token_expiry
from onyx.server.features.mcp.api import _decode_jwt_exp
from onyx.server.features.mcp.api import _hydrate_token_with_remaining_lifetime
from onyx.server.features.mcp.api import OnyxOAuthClientProvider


_TIME_TOLERANCE_SECONDS = 5


def _make_test_jwt(claims: dict[str, Any]) -> str:
    """Build an unsigned JWT-shaped string. Signature is a fixed marker; we
    never verify it — the production decoder only reads claim values for
    expiry tracking."""
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        .rstrip(b"=")
        .decode()
    )
    payload = (
        base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    )
    return f"{header}.{payload}.notasignature"


class TestDecodeJwtExp:
    def test_decodes_exp_claim_from_valid_jwt(self) -> None:
        token = _make_test_jwt({"sub": "user", "exp": 1_900_000_000})

        assert _decode_jwt_exp(token) == 1_900_000_000

    def test_returns_none_for_non_jwt(self) -> None:
        assert _decode_jwt_exp("opaque-random-string") is None

    def test_returns_none_for_jwt_without_exp_claim(self) -> None:
        token = _make_test_jwt({"sub": "user", "scope": "openid"})

        assert _decode_jwt_exp(token) is None

    def test_returns_none_for_malformed_payload_segment(self) -> None:
        # Looks like JWT (3 segments) but middle segment isn't valid base64 JSON
        assert _decode_jwt_exp("header.@@@@.signature") is None

    def test_handles_jwt_segment_needing_padding(self) -> None:
        # The base64url payload below is one byte short of a multiple of 4;
        # decoders that don't re-pad will choke on it.
        claims = {"exp": 1_800_000_000, "x": "a"}
        token = _make_test_jwt(claims)
        # Sanity-check our fixture actually exercises the padding path
        segments = token.split(".")
        assert len(segments[1]) % 4 != 0

        assert _decode_jwt_exp(token) == 1_800_000_000


class TestComputeAbsoluteTokenExpiry:
    def test_uses_expires_in_when_present(self) -> None:
        token = OAuthToken(access_token="opaque", expires_in=3600)

        before = time.time()
        result = _compute_absolute_token_expiry(token)
        after = time.time()

        assert result is not None
        assert before + 3600 - 1 <= result <= after + 3600 + 1

    def test_falls_back_to_jwt_exp_when_expires_in_missing(self) -> None:
        # Salesforce's MCP token response omits `expires_in` entirely; the
        # only way to know lifetime is to read the JWT `exp` claim.
        jwt = _make_test_jwt({"exp": 1_900_000_000})
        token = OAuthToken(access_token=jwt, expires_in=None)

        assert _compute_absolute_token_expiry(token) == 1_900_000_000.0

    def test_returns_none_when_no_expiry_signal(self) -> None:
        token = OAuthToken(access_token="opaque-not-a-jwt", expires_in=None)

        assert _compute_absolute_token_expiry(token) is None

    def test_prefers_expires_in_over_jwt_exp(self) -> None:
        # JWT `exp` may not match the actual access-token validity window
        # the IdP enforces; when the IdP also gives us `expires_in`, trust
        # that as the authoritative remaining lifetime.
        jwt = _make_test_jwt({"exp": 1_900_000_000})
        token = OAuthToken(access_token=jwt, expires_in=60)

        before = time.time()
        result = _compute_absolute_token_expiry(token)

        assert result is not None
        assert before + 60 - 1 <= result <= before + 60 + 2


class TestHydrateTokenWithRemainingLifetime:
    def test_recomputes_expires_in_as_remaining_seconds(self) -> None:
        # Token persisted with abs_expiry 100s from now should hydrate with
        # expires_in roughly equal to 100 (relative-to-now).
        abs_expiry = time.time() + 100
        payload = {
            "access_token": "abc",
            "token_type": "Bearer",
            "expires_in": 999,  # stale issuance-time value; should be overwritten
            _ABS_EXPIRY_KEY: abs_expiry,
        }

        hydrated = _hydrate_token_with_remaining_lifetime(payload)

        assert hydrated.expires_in is not None
        assert 100 - _TIME_TOLERANCE_SECONDS <= hydrated.expires_in <= 100

    def test_returns_negative_expires_in_for_past_expiry(self) -> None:
        # Tokens expired in the past must NOT be reported as valid. The SDK's
        # `is_token_valid()` returns False precisely when `token_expiry_time`
        # is a past timestamp, which `update_token_expiry` produces from a
        # negative `expires_in`.
        abs_expiry = time.time() - 60
        payload = {
            "access_token": "expired",
            "token_type": "Bearer",
            _ABS_EXPIRY_KEY: abs_expiry,
        }

        hydrated = _hydrate_token_with_remaining_lifetime(payload)

        assert hydrated.expires_in is not None
        assert hydrated.expires_in < 0

    def test_payload_without_abs_expiry_passes_through_unchanged(self) -> None:
        # Older persisted rows lack the absolute-expiry stamp; we must not
        # crash and must preserve the original `expires_in` (degrades to
        # pre-fix behavior rather than breaking auth outright).
        payload = {
            "access_token": "abc",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        hydrated = _hydrate_token_with_remaining_lifetime(payload)

        assert hydrated.expires_in == 3600
        assert hydrated.access_token == "abc"

    def test_does_not_mutate_input_payload(self) -> None:
        # Storage code paths read the stored config dict and pass it in;
        # mutating that dict would corrupt the in-memory connection config.
        abs_expiry = time.time() + 10
        payload = {
            "access_token": "abc",
            "token_type": "Bearer",
            _ABS_EXPIRY_KEY: abs_expiry,
        }
        original = dict(payload)

        _hydrate_token_with_remaining_lifetime(payload)

        assert payload == original


class TestAttachAbsoluteExpiry:
    def test_stamps_payload_when_expiry_is_known(self) -> None:
        token = OAuthToken(access_token="abc", expires_in=600)
        payload: dict[str, Any] = token.model_dump(mode="json")

        stamped = _attach_absolute_expiry(payload, token)

        assert _ABS_EXPIRY_KEY in stamped
        assert stamped[_ABS_EXPIRY_KEY] >= time.time() + 600 - _TIME_TOLERANCE_SECONDS

    def test_does_not_stamp_when_expiry_is_unknown(self) -> None:
        token = OAuthToken(access_token="opaque", expires_in=None)
        payload: dict[str, Any] = token.model_dump(mode="json")

        stamped = _attach_absolute_expiry(payload, token)

        assert _ABS_EXPIRY_KEY not in stamped


class TestAttachThenHydrateRoundTrip:
    def test_lifetime_survives_attach_hydrate_roundtrip(self) -> None:
        token = OAuthToken(access_token="abc", expires_in=120)
        payload = _attach_absolute_expiry(token.model_dump(mode="json"), token)

        hydrated = _hydrate_token_with_remaining_lifetime(payload)

        assert hydrated.expires_in is not None
        assert 120 - _TIME_TOLERANCE_SECONDS <= hydrated.expires_in <= 120


class _FakeTokenStorage:
    """Minimal in-memory TokenStorage stand-in for OnyxOAuthClientProvider
    initialization tests.

    Implementing the protocol structurally (not via inheritance) keeps the
    test independent of the SDK's storage-class layout, which has shifted
    across versions.
    """

    def __init__(
        self,
        token: OAuthToken | None,
        client_info: OAuthClientInformationFull | None = None,
    ) -> None:
        self._token = token
        self._client_info = client_info

    async def get_tokens(self) -> OAuthToken | None:
        return self._token

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._token = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, info: OAuthClientInformationFull) -> None:
        self._client_info = info


def _make_provider(
    storage: _FakeTokenStorage,
) -> OnyxOAuthClientProvider:
    async def _redirect(url: str) -> None:  # pragma: no cover - never called
        raise AssertionError("redirect_handler should not be invoked in _initialize")

    async def _callback() -> tuple[str, str | None]:  # pragma: no cover
        raise AssertionError("callback_handler should not be invoked in _initialize")

    return OnyxOAuthClientProvider(
        server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(
            client_name="Onyx Test",
            redirect_uris=[AnyUrl("https://example.com/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=storage,  # type: ignore[arg-type]
        redirect_handler=_redirect,
        callback_handler=_callback,
    )


class TestOnyxOAuthClientProviderInitialize:
    """Tests for `OnyxOAuthClientProvider._initialize`.

    Test bodies drive the async method via `asyncio.run` rather than
    `pytest.mark.asyncio` so the suite doesn't depend on `pytest-asyncio`
    (which isn't installed in this environment).
    """

    def test_seeds_token_expiry_time_for_loaded_tokens(self) -> None:
        # Token has 10s of remaining lifetime (i.e. previously hydrated by
        # OnyxTokenStorage). After _initialize, the SDK's in-memory
        # `token_expiry_time` must reflect that — without this seeding, the
        # SDK treats the token as valid forever and never attempts refresh.
        storage = _FakeTokenStorage(
            token=OAuthToken(access_token="abc", expires_in=10),
        )
        provider = _make_provider(storage)

        before = time.time()
        asyncio.run(provider._initialize())

        assert provider.context.token_expiry_time is not None
        assert (
            before + 10 - _TIME_TOLERANCE_SECONDS
            <= provider.context.token_expiry_time
            <= before + 10 + _TIME_TOLERANCE_SECONDS
        )

    def test_seeds_past_expiry_when_token_is_already_expired(self) -> None:
        # `OnyxTokenStorage.get_tokens` returns negative `expires_in` for
        # tokens whose absolute expiry is in the past. The seeded
        # `token_expiry_time` must end up in the past so the SDK's
        # `is_token_valid()` returns False and the refresh branch fires.
        storage = _FakeTokenStorage(
            token=OAuthToken(access_token="expired", expires_in=-300),
        )
        provider = _make_provider(storage)

        before = time.time()
        asyncio.run(provider._initialize())

        assert provider.context.token_expiry_time is not None
        assert provider.context.token_expiry_time < before

    def test_no_op_when_storage_has_no_tokens(self) -> None:
        storage = _FakeTokenStorage(token=None)
        provider = _make_provider(storage)

        asyncio.run(provider._initialize())

        assert provider.context.current_tokens is None
        assert provider.context.token_expiry_time is None

    def test_no_expiry_set_when_token_has_no_expires_in(self) -> None:
        # An OAuthToken with `expires_in=None` from storage means "no
        # expiry information available" — the seeding must not invent one,
        # so `is_token_valid()` keeps returning True (pre-fix behavior) for
        # this degraded case.
        storage = _FakeTokenStorage(
            token=OAuthToken(access_token="abc", expires_in=None),
        )
        provider = _make_provider(storage)

        asyncio.run(provider._initialize())

        assert provider.context.current_tokens is not None
        assert provider.context.token_expiry_time is None

    def test_is_subclass_of_oauth_client_provider(self) -> None:
        # Sanity check: callers (e.g. `initialize_mcp_client`'s `auth`
        # parameter) are typed against `OAuthClientProvider`; the subclass
        # must remain assignment-compatible.
        storage = _FakeTokenStorage(token=None)
        provider = _make_provider(storage)

        assert isinstance(provider, OAuthClientProvider)
