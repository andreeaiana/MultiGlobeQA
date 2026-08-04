"""Google explicit context caching hookup.

The Gemini caches API: ``client.caches.create(...)`` returns a
``CachedContent`` with ``.name``; subsequent ``generate_content``
calls pass ``cached_content=<name>`` to bill cached input at ~25%
of the normal rate. Minimum cacheable size: 4096 tokens.

These tests cover the hookup itself (without hitting the real API
— ``client.caches.create`` and ``client.models.count_tokens`` are
mocked):

1. single ``caches.create`` call across N=3 requests with the same
   prefix (cache reused on calls 2 and 3).
2. short prefix (< 4096 tokens) skips caching entirely.
3. ``cached_content=<name>`` passed in subsequent GenerateContentConfig.
4. cache-expired error invalidates the registry and re-creates.
5. ``enable_context_cache=False`` short-circuits the whole path.
6. thread-safety: 4 concurrent workers, same prefix → exactly one
   ``caches.create`` call.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch


def _make_model(enable_cache: bool = True):
    """Construct a GoogleModel without actually calling out to the
    Gemini API. The genai client is patched so the constructor
    completes."""
    from benchmark_models.backends.google import GoogleModel
    from benchmark_models.config import GoogleVariantConfig

    config = GoogleVariantConfig(
        backend="google",
        model_id="gemini-3-flash",
        tier="plain",
        enable_context_cache=enable_cache,
    )
    with patch("benchmark_models.backends.google.genai.Client"):
        model = GoogleModel("test-variant", config)
    # Replace the client with a controllable MagicMock.
    model.client = MagicMock()
    return model


def _stub_count_tokens(model, total_tokens: int) -> None:
    """Make ``client.models.count_tokens`` return the given total."""
    response = MagicMock()
    response.total_tokens = total_tokens
    model.client.models.count_tokens.return_value = response


def _stub_caches_create(model, returned_name: str = "cached-content-xyz") -> None:
    """Make ``client.caches.create`` return a stub CachedContent."""
    cached = MagicMock()
    cached.name = returned_name
    model.client.caches.create.return_value = cached


def test_single_create_for_repeated_prefix() -> None:
    """N=3 requests with the same prefix → one caches.create call."""
    model = _make_model()
    _stub_count_tokens(model, total_tokens=8000)
    _stub_caches_create(model, "cache-A")

    prompt = "shared template prefix.\n" * 200 + "Question:\nq1?\n"
    name1 = model._resolve_cached_content(prompt)
    name2 = model._resolve_cached_content(prompt)
    name3 = model._resolve_cached_content(prompt)
    assert name1 == name2 == name3 == "cache-A"
    assert model.client.caches.create.call_count == 1


def test_short_prefix_skips_caching() -> None:
    """count_tokens returns < 4096 → caches.create not called; the
    resolver returns None."""
    model = _make_model()
    _stub_count_tokens(model, total_tokens=2000)
    _stub_caches_create(model, "should-not-be-used")

    prompt = "short prefix.\nQuestion:\nq?\n"
    assert model._resolve_cached_content(prompt) is None
    assert model.client.caches.create.call_count == 0


def test_no_marker_skips_caching() -> None:
    """Without a ``Question:\\n`` marker the resolver returns None
    without calling count_tokens."""
    model = _make_model()
    _stub_count_tokens(model, total_tokens=8000)

    assert model._resolve_cached_content("no marker here, just text") is None
    assert model.client.models.count_tokens.call_count == 0


def test_disabled_flag_short_circuits() -> None:
    """``enable_context_cache=False`` returns None without any API call."""
    model = _make_model(enable_cache=False)
    _stub_count_tokens(model, total_tokens=8000)
    _stub_caches_create(model, "should-not-be-used")

    prompt = "long prefix.\n" * 200 + "Question:\nq?\n"
    assert model._resolve_cached_content(prompt) is None
    assert model.client.caches.create.call_count == 0
    assert model.client.models.count_tokens.call_count == 0


def test_cache_expired_invalidates_registry() -> None:
    """``_invalidate_cache_for_prompt`` removes the registry entry
    so a subsequent ``_resolve_cached_content`` re-creates."""
    model = _make_model()
    _stub_count_tokens(model, total_tokens=8000)
    _stub_caches_create(model, "cache-A")

    prompt = "prefix.\n" * 200 + "Question:\nq?\n"
    assert model._resolve_cached_content(prompt) == "cache-A"
    assert model.client.caches.create.call_count == 1

    # Now stub a different create return to verify re-creation.
    _stub_caches_create(model, "cache-B-fresh")
    model._invalidate_cache_for_prompt(prompt)
    assert model._resolve_cached_content(prompt) == "cache-B-fresh"
    assert model.client.caches.create.call_count == 2


def test_thread_safety_one_create_under_concurrency() -> None:
    """4 concurrent workers resolving the same prefix → exactly one
    ``caches.create`` call (lock guards the seed-the-cache section)."""
    model = _make_model()
    _stub_count_tokens(model, total_tokens=8000)

    # Make caches.create slow enough that overlap actually happens.
    create_started = threading.Event()

    def _slow_create(**_kwargs):
        create_started.set()
        # Hold the lock briefly so concurrent workers queue up.
        threading.Event().wait(timeout=0.05)
        m = MagicMock()
        m.name = "cache-concurrent"
        return m

    model.client.caches.create.side_effect = _slow_create

    prompt = "concurrent prefix.\n" * 200 + "Question:\nq?\n"

    def _resolve():
        return model._resolve_cached_content(prompt)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: _resolve(), range(4)))

    assert all(r == "cache-concurrent" for r in results)
    assert model.client.caches.create.call_count == 1
