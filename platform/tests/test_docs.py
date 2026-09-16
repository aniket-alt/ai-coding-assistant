"""Tests for document generation."""

from dsfs import docs


def test_formats():
    assert docs.fmt(0.8137, "pct") == "81.4%"
    assert docs.fmt(81.377, "pct_raw") == "81.4%"
    assert docs.fmt(-0.0005, "signed4") == "−0.0005"
    assert docs.fmt(None) == "—"


def test_marker_block_is_replaced_and_idempotent():
    text = "a\n<!-- dsfs:x -->\nold\n<!-- /dsfs:x -->\nb"
    once = docs.replace_block(text, "x", "new")
    assert "old" not in once and "new" in once
    assert docs.replace_block(once, "x", "new") == once


def test_missing_marker_leaves_text_untouched():
    assert docs.replace_block("plain", "x", "new") == "plain"
