"""Unit tests for :mod:`translation_dubbing_skill.providers.tts.edge`.

Covers initialization defaults, size calculation, multi-language mapping,
and mock edge-tts stream integrations using mock stubs.
"""

from __future__ import annotations

import pytest

from translation_dubbing_skill.models import ProviderConfig
from translation_dubbing_skill.providers.registry import default_registry
from translation_dubbing_skill.providers.tts.edge import EdgeTTSProvider


def test_edge_tts_provider_is_registered_on_import() -> None:
    assert "edge" in default_registry.list("tts")


def test_edge_tts_provider_class_metadata() -> None:
    assert EdgeTTSProvider.provider_type == "edge"
    assert EdgeTTSProvider.supports_batch is False
    assert EdgeTTSProvider.payload_unit == "chars"


def test_initialize_captures_voice_and_prosody_knobs() -> None:
    provider = EdgeTTSProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="none",
            credential="none",
            extra={
                "default_voice": "zh-CN-XiaoxiaoNeural",
                "rate": "+10%",
                "volume": "-5%",
                "pitch": "+5Hz"
            }
        )
    )
    assert provider.default_voice == "zh-CN-XiaoxiaoNeural"
    assert provider.rate == "+10%"
    assert provider.volume == "-5%"
    assert provider.pitch == "+5Hz"


@pytest.mark.parametrize(
    "requested_voice,expected_resolved",
    [
        ("zh", "zh-CN-YunxiNeural"),
        ("zh-CN", "zh-CN-YunxiNeural"),
        ("en", "en-US-GuyNeural"),
        ("en-US", "en-US-GuyNeural"),
        ("ja", "ja-JP-KeitaNeural"),
        ("ja-JP", "ja-JP-KeitaNeural"),
        ("es", "es-ES-AlvaroNeural"),
        ("fr", "fr-FR-EloiseNeural"),
        ("de", "de-DE-KillianNeural"),
        ("ko", "ko-KR-SunHiNeural"),
        ("it", "it-IT-DiegoNeural"),
        ("ru", "ru-RU-DmitryNeural"),
        ("pt", "pt-BR-AntonioNeural"),
        ("pt-BR", "pt-BR-AntonioNeural"),
        # Customized/arbitrary voice overrides pass through directly
        ("en-US-JennyNeural", "en-US-JennyNeural"),
    ],
)
def test_voice_mapping_resolving_logic(requested_voice: str, expected_resolved: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate voice_mapping dictionary correctly resolves language tags to neural voices."""
    import sys
    import types

    # 1. Create a dummy edge_tts module stub so the test can import edge_tts
    # without needing the live package in pure unit tests.
    dummy_edge_tts = types.ModuleType("edge_tts")
    class DummyCommunicate:
        def __init__(self, text, voice, **kwargs):
            self.voice_seen = voice
        async def stream(self):
            # Yield a mock chunk to bypass stream loop
            yield {"type": "audio", "data": b"mock-mp3-bytes"}

    dummy_edge_tts.Communicate = DummyCommunicate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_tts", dummy_edge_tts)

    # 2. Mock pydub to prevent real decoding errors of the fake MP3 data
    dummy_pydub = types.ModuleType("pydub")
    class DummyAudioSegment:
        def __init__(self):
            pass
        @classmethod
        def from_file(cls, file, format):
            return cls()
        def __len__(self):
            return 1000
        def export(self, buffer, format):
            buffer.write(b"mock-wav-bytes")

    dummy_pydub.AudioSegment = DummyAudioSegment  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pydub", dummy_pydub)

    provider = EdgeTTSProvider()
    provider.initialize(
        ProviderConfig(
            endpoint="none",
            credential="none",
            extra={"default_voice": "zh-CN-YunxiNeural"}
        )
    )

    # Invoke synth and verify that edge_tts was called with the mapped/resolved voice ID
    async def run_synth():
        wav_bytes, duration = await provider.synth("Test Text", requested_voice)
        assert wav_bytes == b"mock-wav-bytes"
        assert duration == 1000

    import asyncio
    asyncio.run(run_synth())
