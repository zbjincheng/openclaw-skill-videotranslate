"""Property-based tests for the translation layer (P6, P7, P12, P25).

Covers tasks 6.4, 6.5, 6.6 and 6.7 of the spec. Each block below
implements one property from ``design.md``:

- P6  — 翻译提供方结构契约 (R5.1, R5.2, R7.1, R7.5)
- P7  — 翻译文本语义契约 (R5.6, R7.2)
- P12 — 提供方契约违例检测 (R7.6)
- P25 — 翻译与 TTS 进度事件完整性（翻译部分）(R11.2)

All provider-contract tests (P6, P7) are parametrised over both
``LLMTranslationProvider`` and ``WebTranslationProvider``; network
traffic is stubbed via :class:`httpx.MockTransport` so the tests run
fully offline. P12 drives the coordinator with an in-memory contract-
breaking mock that deliberately violates structural/semantic contracts
so the property can assert the resulting
:class:`ProviderContractViolationError` carries ``violated_clause`` and
``provider_type``. P25 uses an :class:`_InMemoryReporter` to capture
the translator's progress events and asserts monotonicity, final
value, and the ``total`` field on every event.
"""

from __future__ import annotations

import json
from typing import Any, Callable, ClassVar

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from translation_dubbing_skill.errors import ProviderContractViolationError
from translation_dubbing_skill.models import (
    ProgressEvent,
    ProviderConfig,
    SubtitleEntry,
)
from translation_dubbing_skill.providers import ProviderRegistry
from translation_dubbing_skill.providers.translation.llm import (
    LLMTranslationProvider,
)
from translation_dubbing_skill.providers.translation.web import (
    WebTranslationProvider,
)
from translation_dubbing_skill.scheduler import ProviderRateLimitConfig
from translation_dubbing_skill.translation import Translator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_rate_limit_config(**overrides: Any) -> ProviderRateLimitConfig:
    """Return a permissive config (large caps, no probing) with overrides."""
    base: dict[str, Any] = dict(
        batch_size_initial=4,
        batch_size_min=1,
        batch_size_max=16,
        payload_size_initial=10_000,
        payload_size_min=10,
        payload_size_max=100_000,
        payload_unit="chars",
        concurrency_initial=2,
        concurrency_min=1,
        concurrency_max=8,
        max_retries=1,
        backoff_base_ms=1,
        backoff_jitter_ms=0,
        probe_up_every_n_success=10_000,
        supports_batch=True,
    )
    base.update(overrides)
    return ProviderRateLimitConfig(**base)


class _InMemoryReporter:
    """Captures :class:`ProgressEvent` objects for later assertion."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def report(self, event: ProgressEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Hypothesis generators
# ---------------------------------------------------------------------------


# Source text deliberately excludes whitespace-only strings so the semantic
# contract path "non-empty input → non-empty Chinese output" can be
# exercised without ambiguity. It keeps the alphabet latin + digits so the
# mock transports can safely embed the id in the output.
_NONEMPTY_TEXT = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
    ),
    min_size=1,
    max_size=20,
)

# Generator allowing both empty/whitespace and non-empty text so P6/P7
# cover the mixed case the translator actually sees.
_MIXED_TEXT = st.one_of(
    st.sampled_from(["", " ", "  ", "\t", "\n"]),
    _NONEMPTY_TEXT,
)


@st.composite
def _subtitle_entries(
    draw: st.DrawFn,
    *,
    min_size: int = 1,
    max_size: int = 6,
    text_strategy: st.SearchStrategy[str] = _MIXED_TEXT,
) -> list[SubtitleEntry]:
    """Draw a list of non-overlapping subtitle entries with monotonic times.

    Each entry's index is 1-based and strictly increasing; start/end
    timestamps are positive and ``end_ms > start_ms`` is enforced.
    """
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    entries: list[SubtitleEntry] = []
    cursor = 0
    for i in range(count):
        start = cursor + draw(st.integers(min_value=0, max_value=100))
        duration = draw(st.integers(min_value=1, max_value=1_000))
        end = start + duration
        text = draw(text_strategy)
        entries.append(
            SubtitleEntry(
                index=i + 1,
                start_ms=start,
                end_ms=end,
                text=text,
            )
        )
        cursor = end
    return entries


# ---------------------------------------------------------------------------
# Mock-transport handlers for real providers (P6, P7)
# ---------------------------------------------------------------------------


def _llm_handler(request: httpx.Request) -> httpx.Response:
    """Mock the chat-completions endpoint used by ``LLMTranslationProvider``.

    The provider sends a ``system`` message with instructions and a
    ``user`` message containing a JSON array of ``{"id", "text"}``
    entries. We grep the JSON array out of whichever message carries
    it (either message is allowed to evolve), translate each item to a
    deterministic ``"中文-<id>"`` — guaranteed non-empty simplified
    Chinese — and return a chat-completions-shaped response.
    """
    body = json.loads(request.content)
    # Find the first JSON array that looks like the translation input.
    array_text: str | None = None
    for message in body.get("messages", []):
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        start = content.find("[")
        end = content.rfind("]")
        if start == -1 or end == -1 or end <= start:
            continue
        try:
            candidate = json.loads(content[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(candidate, list) and all(
            isinstance(item, dict) and "id" in item for item in candidate
        ):
            array_text = content[start : end + 1]
            break
    if array_text is None:
        # Defensive: if the caller's prompt ever drops the array we bail
        # with a clear assertion instead of a cryptic KeyError.
        raise AssertionError(
            "LLM mock handler could not find a translation input array in: "
            + repr(body)
        )
    items = json.loads(array_text)
    translations = [
        {"id": item["id"], "translation": f"中文-{item['id']}"} for item in items
    ]
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": json.dumps(translations, ensure_ascii=False)}}
            ]
        },
    )


def _web_handler(request: httpx.Request) -> httpx.Response:
    """Mock a generic REST ``/translate`` endpoint for ``WebTranslationProvider``.

    Echoes a deterministic ``"中文-<q>"`` translation so the output is
    always non-empty simplified Chinese.
    """
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={"translatedText": f"中文-{body['q']}"},
    )


def _make_llm_provider() -> LLMTranslationProvider:
    """Build an initialised LLM provider wired to :func:`_llm_handler`."""
    provider = LLMTranslationProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://api.llm.test/v1/chat/completions",
            credential="secret",
            extra={"model_name": "gpt-test"},
        )
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_llm_handler)
    )
    return provider


def _make_web_provider() -> WebTranslationProvider:
    """Build an initialised Web provider wired to :func:`_web_handler`."""
    provider = WebTranslationProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="https://api.web.test/translate",
            credential="secret",
            extra={"language_pair": "en-zh"},
        )
    )
    provider._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_web_handler)
    )
    return provider


_PROVIDER_FACTORIES: list[tuple[str, Callable[[], Any]]] = [
    ("llm", _make_llm_provider),
    ("web", _make_web_provider),
]


# ---------------------------------------------------------------------------
# Property 6 — 翻译提供方结构契约 (task 6.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_label, make_provider",
    _PROVIDER_FACTORIES,
    ids=[label for label, _ in _PROVIDER_FACTORIES],
)
@given(entries=_subtitle_entries())
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_property_6_translation_provider_structural_contract(
    provider_label: str,
    make_provider: Callable[[], Any],
    entries: list[SubtitleEntry],
) -> None:
    """**Validates: Requirements 5.1, 5.2, 7.1, 7.5**

    For any registered translation provider and any legal list of subtitle
    entries, the provider's output SHALL preserve length, index,
    start_ms, and end_ms per position.
    """
    provider = make_provider()
    try:
        output = await provider.translate_batch(entries)

        assert len(output) == len(entries), (
            f"{provider_label}: expected len {len(entries)}, got {len(output)}"
        )
        for src, dst in zip(entries, output):
            assert dst.index == src.index
            assert dst.start_ms == src.start_ms
            assert dst.end_ms == src.end_ms
    finally:
        await provider.aclose()


# ---------------------------------------------------------------------------
# Property 7 — 翻译文本语义契约 (task 6.5)
# ---------------------------------------------------------------------------


# CJK Unified Ideographs range — every Han character in the mock's
# ``"中文-<id>"`` output lives here, so a simple scan suffices.
def _contains_cjk(text: str) -> bool:
    return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)


@pytest.mark.parametrize(
    "provider_label, make_provider",
    _PROVIDER_FACTORIES,
    ids=[label for label, _ in _PROVIDER_FACTORIES],
)
@given(entries=_subtitle_entries())
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_property_7_translation_semantic_contract(
    provider_label: str,
    make_provider: Callable[[], Any],
    entries: list[SubtitleEntry],
) -> None:
    """**Validates: Requirements 5.6, 7.2**

    Blank input (empty or whitespace-only) maps to the empty string;
    non-blank input maps to non-empty simplified Chinese.

    The coordinator (``Translator``) is responsible for the blank
    fast-path — providers never see blank text. Routing through the
    ``Translator`` is therefore the correct point of observation here.
    """
    provider = make_provider()
    registry = ProviderRegistry()
    # Inject the pre-initialised provider so the coordinator's own
    # construction path is bypassed (the real one would try another
    # ``initialize`` call and re-arm the mock transport pointlessly).
    registry.register(
        "translation", provider_label, _PreInitialisedProvider.factory(provider)
    )
    translator = Translator(registry)
    config = ProviderConfig(endpoint="https://stub", credential="stub")

    try:
        output = await translator.translate(
            entries=entries,
            provider_type=provider_label,
            config=config,
            rate_limit_config=_make_rate_limit_config(),
        )

        assert len(output) == len(entries)
        for src, dst in zip(entries, output):
            if src.text.strip() == "":
                assert dst.text == "", (
                    f"{provider_label}: blank input produced non-empty "
                    f"output {dst.text!r}"
                )
            else:
                assert dst.text != "", (
                    f"{provider_label}: non-blank input produced empty output"
                )
                assert _contains_cjk(dst.text), (
                    f"{provider_label}: output {dst.text!r} has no CJK chars"
                )
    finally:
        await provider.aclose()


class _PreInitialisedProvider:
    """Adapter that lets the registry reuse an already-constructed provider.

    ``ProviderRegistry.create`` instantiates ``cls()`` then calls
    ``initialize(config)`` on the result. Our P6/P7 tests build real
    providers with a mock HTTP transport before calling the coordinator;
    that shared instance is what the property needs to reach. The class
    method :meth:`factory` produces a zero-arg class closing over the
    pre-built instance so the registry's ``cls()`` returns it unchanged
    and the extra ``initialize`` call is a no-op.
    """

    @classmethod
    def factory(cls, provider: Any) -> type:
        class _Bound:
            provider_type = getattr(provider, "provider_type", "mock")
            supports_batch = getattr(provider, "supports_batch", True)
            payload_unit = getattr(provider, "payload_unit", "chars")

            def __init__(self) -> None:
                self._inner = provider

            def initialize(self, config: ProviderConfig) -> None:
                # Already initialised — don't disturb the mock transport.
                pass

            def size_of(self, text: str) -> int:
                sizer = getattr(self._inner, "size_of", None)
                if callable(sizer):
                    return sizer(text)
                return len(text)

            async def translate_batch(
                self,
                entries: list[SubtitleEntry],
                target_language: str = "zh-CN",
            ) -> list[SubtitleEntry]:
                return await self._inner.translate_batch(entries, target_language)

            async def translate(
                self,
                entries: list[SubtitleEntry],
                target_language: str = "zh-CN",
            ) -> list[SubtitleEntry]:
                return await self._inner.translate_batch(entries, target_language)

        return _Bound


# ---------------------------------------------------------------------------
# Property 12 — 提供方契约违例检测 (task 6.6)
# ---------------------------------------------------------------------------


_VIOLATION_MODES: tuple[str, ...] = (
    "length",
    "index",
    "start_ms",
    "end_ms",
    "empty_output",
    "non_chinese",
)


class _ContractBreakingProvider:
    """Mock provider whose ``translate_batch`` deliberately violates contracts.

    The ``mode`` class variable selects the violation kind; tests set
    it on the class before calling the coordinator because
    ``ProviderRegistry.create`` builds fresh instances per call.
    """

    provider_type: ClassVar[str] = "violator"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"
    mode: ClassVar[str] = "length"

    def initialize(self, config: ProviderConfig) -> None:
        pass

    def size_of(self, text: str) -> int:
        return len(text)

    async def translate_batch(
        self,
        entries: list[SubtitleEntry],
        target_language: str = "zh-CN",
    ) -> list[SubtitleEntry]:
        if self.mode == "length":
            # Return one fewer entry — scheduler catches length mismatch.
            return [
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    text="中文",
                )
                for e in entries[:-1]
            ]
        if self.mode == "index":
            return [
                SubtitleEntry(
                    index=e.index + 1_000,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    text="中文",
                )
                for e in entries
            ]
        if self.mode == "start_ms":
            return [
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms + 7,
                    end_ms=e.end_ms,
                    text="中文",
                )
                for e in entries
            ]
        if self.mode == "end_ms":
            return [
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms + 7,
                    text="中文",
                )
                for e in entries
            ]
        if self.mode == "empty_output":
            return [
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    text="",  # non-empty input → empty output
                )
                for e in entries
            ]
        if self.mode == "non_chinese":
            return [
                SubtitleEntry(
                    index=e.index,
                    start_ms=e.start_ms,
                    end_ms=e.end_ms,
                    text="latin only",
                )
                for e in entries
            ]
        raise AssertionError(f"unknown violation mode: {self.mode}")


@given(
    entries=_subtitle_entries(
        min_size=2,  # need >= 2 for length-mismatch to remain non-empty
        max_size=5,
        text_strategy=_NONEMPTY_TEXT,  # avoid blank fast-path
    ),
    mode=st.sampled_from(_VIOLATION_MODES),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_property_12_contract_violation_detection(
    entries: list[SubtitleEntry],
    mode: str,
) -> None:
    """**Validates: Requirement 7.6**

    A provider that returns a malformed response must cause the
    translation coordinator to raise either:

    - :class:`ProviderContractViolationError` directly (structural +
      semantic mismatches caught by the coordinator's contract check),
      or
    - A :class:`TranslationError` whose chain contains a
      ``ProviderContractViolationError``-adjacent description (for the
      length-mismatch path, where the scheduler itself detects the
      malformed response and bubbles up as retry exhaustion; the
      coordinator wraps that in a ``TranslationError`` that still
      carries ``provider_type``). Both forms satisfy R7.6's "violated
      contract is surfaced with provider_type".
    """
    registry = ProviderRegistry()
    registry.register("translation", "violator", _ContractBreakingProvider)
    _ContractBreakingProvider.mode = mode  # type: ignore[misc]
    translator = Translator(registry)

    with pytest.raises(Exception) as excinfo:
        await translator.translate(
            entries=entries,
            provider_type="violator",
            config=ProviderConfig(endpoint="stub", credential="stub"),
            rate_limit_config=_make_rate_limit_config(
                max_retries=0,  # fail fast on length mismatch
                # Keep every batch a single call so the mismatch from the
                # `length` mode is observed in full (the scheduler treats
                # the short response as `SchedulerBatchFailure`).
                batch_size_initial=len(entries),
                batch_size_max=len(entries),
            ),
        )

    err = excinfo.value

    # For modes the coordinator itself checks, the raised type must be
    # ``ProviderContractViolationError`` and ``violated_clause`` and
    # ``provider_type`` must be present in the context.
    if mode != "length":
        assert isinstance(err, ProviderContractViolationError), (
            f"mode={mode}: expected ProviderContractViolationError, "
            f"got {type(err).__name__}"
        )
        assert "violated_clause" in err.context
        assert err.context["provider_type"] == "violator"
    else:
        # Length mismatch is detected by the scheduler; the coordinator
        # wraps it in TranslationError still carrying provider_type.
        from translation_dubbing_skill.errors import TranslationError

        assert isinstance(err, TranslationError), (
            f"length mismatch should raise TranslationError, "
            f"got {type(err).__name__}"
        )
        assert err.context.get("provider_type") == "violator"


# ---------------------------------------------------------------------------
# Property 25 — 翻译阶段进度事件完整性 (task 6.7)
# ---------------------------------------------------------------------------


class _ProgressProvider:
    """Trivial always-succeeding provider for progress-event observation.

    Returns one output per input, tagged with a deterministic Chinese
    prefix so the coordinator's semantic contract check passes
    regardless of input text.
    """

    provider_type: ClassVar[str] = "progress"
    supports_batch: ClassVar[bool] = True
    payload_unit: ClassVar[str] = "chars"

    def initialize(self, config: ProviderConfig) -> None:
        pass

    def size_of(self, text: str) -> int:
        return len(text)

    async def translate_batch(
        self,
        entries: list[SubtitleEntry],
        target_language: str = "zh-CN",
    ) -> list[SubtitleEntry]:
        return [
            SubtitleEntry(
                index=e.index,
                start_ms=e.start_ms,
                end_ms=e.end_ms,
                text=f"中文{e.index}",
            )
            for e in entries
        ]


@given(
    entries=_subtitle_entries(
        min_size=1,
        max_size=8,
        text_strategy=_NONEMPTY_TEXT,  # ensure len(entries) == non-blank count
    ),
    batch_size=st.integers(min_value=1, max_value=5),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.asyncio
async def test_property_25_translation_progress_event_completeness(
    entries: list[SubtitleEntry],
    batch_size: int,
) -> None:
    """**Validates: Requirement 11.2**

    For any legal list of entries, the translator's progress events
    satisfy:

    - ``completed`` is monotonically non-decreasing,
    - final ``completed`` equals ``len(entries)``,
    - every event reports ``total == len(entries)``.

    All entries in this property are non-blank so
    ``len(entries) == len(non_blank_entries)``; the final ``completed``
    landed by the coordinator therefore coincides with the full input
    length.
    """
    registry = ProviderRegistry()
    registry.register("translation", "progress", _ProgressProvider)
    reporter = _InMemoryReporter()
    translator = Translator(registry, reporter=reporter)

    await translator.translate(
        entries=entries,
        provider_type="progress",
        config=ProviderConfig(endpoint="stub", credential="stub"),
        rate_limit_config=_make_rate_limit_config(
            batch_size_initial=batch_size,
            batch_size_max=max(batch_size, 1),
        ),
    )

    total = len(entries)
    assert reporter.events, "expected at least one progress event"

    completeds: list[int] = []
    for ev in reporter.events:
        assert ev.stage == "translating"
        assert ev.total == total, (
            f"expected total={total} on every event, got {ev.total}"
        )
        assert ev.completed is not None
        completeds.append(ev.completed)

    # Monotonic non-decreasing.
    for a, b in zip(completeds, completeds[1:]):
        assert b >= a, f"completed regressed: {a} -> {b}"

    # Final value equals total.
    assert completeds[-1] == total, (
        f"final completed {completeds[-1]} != total {total}"
    )
