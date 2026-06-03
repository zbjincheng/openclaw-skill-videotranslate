"""Unit tests for text normalization utility."""

from __future__ import annotations

import pytest

from translation_dubbing_skill.tts.text_normalizer import normalize_chinese_text


@pytest.mark.parametrize(
    "input_text,expected",
    [
        # Markdown cleaning
        ("**粗体文本** 和 *斜体*", "粗体文本 和 斜体"),
        ("__下划线__", "下划线"),
        ("[链接文字](链接地址)", "链接文字"),
        # Percentages
        ("增长了 25%", "增长了 百分之二十五"),
        ("100%", "百分之一百"),
        ("0%", "百分之零"),
        # Integers
        ("100", "一百"),
        ("99", "九十九"),
        ("12", "十二"),
        ("10", "十"),
        ("2026", "二千零二十六"),
        ("0", "零"),
        # Combined case
        ("这件商品打了 **9折**，节省了 10%，现价只要 99 元！", "这件商品打了 九折，节省了 百分之十，现价只要 九十九 元！"),
    ],
)
def test_normalize_chinese_text(input_text: str, expected: str) -> None:
    assert normalize_chinese_text(input_text) == expected
