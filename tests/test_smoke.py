"""Smoke test to verify the package scaffold is wired up correctly."""

import translation_dubbing_skill


def test_package_importable() -> None:
    """The top-level package imports and exposes a version string."""
    assert isinstance(translation_dubbing_skill.__version__, str)
    assert translation_dubbing_skill.__version__
