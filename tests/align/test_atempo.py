"""Unit tests for :mod:`translation_dubbing_skill.align.atempo`.

Covers the pure helper :func:`build_atempo_chain` with both example-based
assertions (the canonical decomposition cases from the design document)
and a property test that pins down the ``∈ [0.5, 2.0] per stage``
invariant across a broad range of rates.

``apply_atempo`` is exercised indirectly by the aligner integration path
and therefore not mocked here; these tests stay ffmpeg-free by covering
only the decomposition logic.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from translation_dubbing_skill.align.atempo import build_atempo_chain


def _parse_chain(chain: str) -> list[float]:
    """Extract the per-stage rates from an ``atempo=a,atempo=b,…`` string."""
    assert chain.startswith("atempo="), chain
    stages: list[float] = []
    for stage in chain.split(","):
        assert stage.startswith("atempo="), stage
        stages.append(float(stage.removeprefix("atempo=")))
    return stages


@pytest.mark.parametrize(
    "rate,expected",
    [
        (1.0, "atempo=1.0"),
        (1.5, "atempo=1.5"),
        (2.0, "atempo=2.0"),
        (0.5, "atempo=0.5"),
    ],
)
def test_build_atempo_chain_single_stage(rate: float, expected: str) -> None:
    """Rates already in ``[0.5, 2.0]`` emit a single stage (the rate itself)."""
    assert build_atempo_chain(rate) == expected


def test_build_atempo_chain_3x_is_design_example() -> None:
    """``3.0`` decomposes to ``atempo=2.0,atempo=1.5`` per the design doc."""
    assert build_atempo_chain(3.0) == "atempo=2.0,atempo=1.5"


def test_build_atempo_chain_quarter_is_symmetric() -> None:
    """``0.25`` decomposes to ``atempo=0.5,atempo=0.5``."""
    assert build_atempo_chain(0.25) == "atempo=0.5,atempo=0.5"


def test_build_atempo_chain_rejects_zero() -> None:
    with pytest.raises(ValueError):
        build_atempo_chain(0.0)


def test_build_atempo_chain_rejects_negative() -> None:
    with pytest.raises(ValueError):
        build_atempo_chain(-1.0)


def test_build_atempo_chain_rejects_nan() -> None:
    with pytest.raises(ValueError):
        build_atempo_chain(float("nan"))


@given(rate=st.floats(min_value=0.01, max_value=64.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_build_atempo_chain_stages_within_bounds(rate: float) -> None:
    """Every per-stage factor must land in ffmpeg's ``[0.5, 2.0]`` range.

    The chain's product must also agree with ``rate`` up to floating-point
    precision; otherwise the decomposition would be silently wrong.
    """
    chain = build_atempo_chain(rate)
    stages = _parse_chain(chain)

    for stage in stages:
        assert 0.5 - 1e-9 <= stage <= 2.0 + 1e-9, (
            f"stage {stage!r} outside [0.5, 2.0]; rate={rate!r} chain={chain!r}"
        )

    product = 1.0
    for stage in stages:
        product *= stage
    assert math.isclose(product, rate, rel_tol=1e-6), (
        f"chain product {product!r} != rate {rate!r} (chain={chain!r})"
    )
