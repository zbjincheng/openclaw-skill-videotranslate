"""Smoke tests for the OpenClaw ``manifest.yaml`` (task 14.6).

These tests validate the skill's manifest against the contract described
in design.md §"清单文件 Schema". They do not exercise the Python parser
(``entry.manifest.parse_manifest``) — they verify the *YAML document
itself* declares every required field, every required default, and every
enum value the runtime and the entry point rely on.

The reasoning: ``parse_manifest`` only runs after the OpenClaw runtime
has already interpreted the manifest. If the manifest itself drifts
from the design (wrong enum, missing default, renamed field), the
Python code never gets the chance to catch it. This smoke test closes
that gap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "manifest.yaml"


@pytest.fixture(scope="module")
def manifest() -> dict:
    """Parse ``manifest.yaml`` once per module."""
    assert MANIFEST_PATH.exists(), f"manifest.yaml missing at {MANIFEST_PATH}"
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Top-level structural checks
# ---------------------------------------------------------------------------


def test_manifest_top_level_fields_present(manifest: dict) -> None:
    """The manifest must declare name, version, inputs, outputs and formats."""
    for key in ("name", "version", "inputs", "outputs", "supported_video_formats"):
        assert key in manifest, f"manifest.yaml missing top-level field: {key}"


def test_supported_video_formats_non_empty(manifest: dict) -> None:
    """R10.3 — supported_video_formats must enumerate at least one format."""
    formats = manifest["supported_video_formats"]
    assert isinstance(formats, list) and formats, (
        "supported_video_formats must be a non-empty list"
    )
    # All entries must be non-empty strings.
    for ext in formats:
        assert isinstance(ext, str) and ext, (
            f"supported_video_formats has a non-string or empty entry: {ext!r}"
        )


def test_outputs_include_video_and_subtitle_paths(manifest: dict) -> None:
    """R9.16 — outputs must include output_video_path and output_subtitle_path."""
    outputs = manifest["outputs"]
    assert "output_video_path" in outputs
    assert "output_subtitle_path" in outputs


# ---------------------------------------------------------------------------
# processing_mode
# ---------------------------------------------------------------------------


def test_processing_mode_enum_and_default(manifest: dict) -> None:
    """R1.3 — processing_mode enum values + default == subtitle_and_dubbing."""
    field = manifest["inputs"]["processing_mode"]
    assert field["type"] == "enum"
    assert set(field["values"]) == {"subtitle_only", "subtitle_and_dubbing"}
    assert field["default"] == "subtitle_and_dubbing"


# ---------------------------------------------------------------------------
# Provider enums
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name",
    ["translation_provider", "tts_provider"],
)
def test_provider_enum_values(manifest: dict, field_name: str) -> None:
    """R1.4 / R1.12 — translation_provider / tts_provider enum == {llm, web}."""
    field = manifest["inputs"][field_name]
    assert field["type"] == "enum"
    assert set(field["values"]) == {"llm", "web"}


def test_tts_provider_required_when_clause(manifest: dict) -> None:
    """R1.6 — tts_provider must be required_when processing_mode == subtitle_and_dubbing."""
    for name in ("tts_provider", "tts_endpoint", "tts_credential"):
        field = manifest["inputs"][name]
        assert "required_when" in field, (
            f"{name} must declare required_when in manifest.yaml"
        )
        clause = field["required_when"]
        assert clause == {"processing_mode": "subtitle_and_dubbing"}, (
            f"{name}.required_when must pin processing_mode to "
            f"subtitle_and_dubbing; got {clause!r}"
        )


# ---------------------------------------------------------------------------
# Rate-limit defaults
# ---------------------------------------------------------------------------


_RATE_LIMIT_REQUIRED_KEYS = {
    "batch_size_initial",
    "batch_size_min",
    "batch_size_max",
    "payload_size_initial",
    "payload_size_min",
    "payload_size_max",
    "payload_unit",
    "concurrency_initial",
    "concurrency_min",
    "concurrency_max",
    "max_retries",
    "backoff_base_ms",
    "backoff_jitter_ms",
    "probe_up_every_n_success",
    "supports_batch",
}


@pytest.mark.parametrize(
    "field_name",
    ["translation_rate_limit", "tts_rate_limit"],
)
def test_rate_limit_default_has_all_required_keys(
    manifest: dict, field_name: str
) -> None:
    """R12.3 / R12.4 — both rate-limit defaults declare every scheduler knob."""
    default = manifest["inputs"][field_name]["default"]
    missing = _RATE_LIMIT_REQUIRED_KEYS - set(default.keys())
    assert not missing, (
        f"{field_name}.default is missing keys: {sorted(missing)}"
    )


@pytest.mark.parametrize(
    "field_name",
    ["translation_rate_limit", "tts_rate_limit"],
)
def test_rate_limit_default_values_are_in_range(
    manifest: dict, field_name: str
) -> None:
    """Defaults satisfy min <= initial <= max on every dimension (R12.3, R12.4)."""
    default = manifest["inputs"][field_name]["default"]

    for prefix in ("batch_size", "payload_size", "concurrency"):
        minimum = default[f"{prefix}_min"]
        initial = default[f"{prefix}_initial"]
        maximum = default[f"{prefix}_max"]
        assert minimum <= initial <= maximum, (
            f"{field_name}.default.{prefix}: min={minimum}, "
            f"initial={initial}, max={maximum} violates min <= initial <= max"
        )
        # Every dimension must allow at least one unit.
        assert minimum >= 1, (
            f"{field_name}.default.{prefix}_min must be >= 1, got {minimum}"
        )

    assert default["payload_unit"] in ("chars", "tokens"), (
        f"{field_name}.default.payload_unit must be chars or tokens; "
        f"got {default['payload_unit']!r}"
    )
    assert default["max_retries"] >= 0
    assert default["backoff_base_ms"] > 0
    assert default["backoff_jitter_ms"] >= 0
    assert default["probe_up_every_n_success"] > 0
    assert isinstance(default["supports_batch"], bool)


def test_translation_rate_limit_supports_batch_default(manifest: dict) -> None:
    """Translation's default supports_batch is True (matches LLM/Web impls)."""
    default = manifest["inputs"]["translation_rate_limit"]["default"]
    assert default["supports_batch"] is True


def test_tts_rate_limit_supports_batch_default(manifest: dict) -> None:
    """TTS's default supports_batch is False (third-party web TTS is single-shot)."""
    default = manifest["inputs"]["tts_rate_limit"]["default"]
    assert default["supports_batch"] is False


# ---------------------------------------------------------------------------
# Cross-check: manifest defaults round-trip through parse_manifest
# ---------------------------------------------------------------------------


def test_manifest_defaults_construct_valid_rate_limit_configs(
    manifest: dict,
) -> None:
    """Loading the defaults via ProviderRateLimitConfig must not raise."""
    from translation_dubbing_skill.scheduler.config import ProviderRateLimitConfig

    for field_name in ("translation_rate_limit", "tts_rate_limit"):
        default = manifest["inputs"][field_name]["default"]
        # ``__post_init__`` enforces the same invariants the scheduler
        # relies on at runtime. If this constructor raises, the manifest
        # default is inconsistent with the Python config's expectations.
        ProviderRateLimitConfig(**default)
