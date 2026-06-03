"""Unit tests for :class:`ProviderConfig`.

Covers requirement R6.2 at the data-model layer:
- Fields ``endpoint``, ``credential``, ``extra`` exist with correct types.
- ``extra`` defaults to an empty dict (not a shared mutable default).
- Values round-trip through construction.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from translation_dubbing_skill.models import ProviderConfig
from translation_dubbing_skill.models import provider_config as pc_module


def test_provider_config_is_dataclass_with_expected_fields() -> None:
    """ProviderConfig is a dataclass with the design-mandated fields."""
    assert is_dataclass(ProviderConfig)
    field_names = [f.name for f in fields(ProviderConfig)]
    assert field_names == ["endpoint", "credential", "extra"]


def test_provider_config_stores_values() -> None:
    """Constructing with explicit values preserves them verbatim."""
    config = ProviderConfig(
        endpoint="https://api.example.com/v1",
        credential="sk-secret",
        extra={"model_name": "gpt-x", "temperature": 0.2},
    )
    assert config.endpoint == "https://api.example.com/v1"
    assert config.credential == "sk-secret"
    assert config.extra == {"model_name": "gpt-x", "temperature": 0.2}


def test_provider_config_extra_defaults_to_empty_dict() -> None:
    """``extra`` defaults to an empty dict when omitted."""
    config = ProviderConfig(endpoint="https://x", credential="c")
    assert config.extra == {}


def test_provider_config_extra_default_is_not_shared() -> None:
    """Each instance gets its own ``extra`` dict (no shared mutable default)."""
    a = ProviderConfig(endpoint="https://x", credential="c")
    b = ProviderConfig(endpoint="https://y", credential="d")
    a.extra["key"] = "value"
    assert b.extra == {}
    assert a.extra is not b.extra


def test_provider_config_equality_is_value_based() -> None:
    """Two configs with identical fields compare equal."""
    a = ProviderConfig(endpoint="https://x", credential="c", extra={"k": 1})
    b = ProviderConfig(endpoint="https://x", credential="c", extra={"k": 1})
    c = ProviderConfig(endpoint="https://x", credential="c", extra={"k": 2})
    assert a == b
    assert a != c


def test_module_exports_public_api() -> None:
    """Public API is re-exported from the models package."""
    assert set(pc_module.__all__) == {"ProviderConfig"}
    from translation_dubbing_skill import models as models_pkg

    assert "ProviderConfig" in models_pkg.__all__
    assert hasattr(models_pkg, "ProviderConfig")
