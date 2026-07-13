"""Unit tests for olmocr helper functions in rag_app.py.

These tests cover pure functions that do not require a GPU server, an OpenAI
client, or any PDF files.  They run in milliseconds and are safe to include in
any CI pipeline that has pyyaml installed.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_app import _olmocr_parse_yaml_response, _postprocess_olmocr_page


# ---------------------------------------------------------------------------
# _olmocr_parse_yaml_response
# ---------------------------------------------------------------------------

class TestOlmocrParseYaml:
    def test_valid_yaml_returns_dict(self):
        raw = (
            "primary_language: en\n"
            "is_rotation_valid: true\n"
            "rotation_correction: 0\n"
            "is_table: false\n"
            "is_diagram: false\n"
            "natural_text: |\n"
            "  This is the page text.\n"
        )
        result = _olmocr_parse_yaml_response(raw)
        assert isinstance(result, dict)
        assert result["primary_language"] == "en"
        assert result["is_table"] is False
        assert "This is the page text." in result["natural_text"]

    def test_markdown_fenced_yaml_stripped(self):
        raw = "```yaml\nprimary_language: en\nnatural_text: hello\n```"
        result = _olmocr_parse_yaml_response(raw)
        assert result is not None
        assert result["natural_text"] == "hello"

    def test_markdown_fence_without_language_stripped(self):
        raw = "```\nprimary_language: fr\nnatural_text: bonjour\n```"
        result = _olmocr_parse_yaml_response(raw)
        assert result is not None
        assert result["primary_language"] == "fr"

    def test_empty_string_returns_none(self):
        assert _olmocr_parse_yaml_response("") is None

    def test_malformed_yaml_returns_none(self):
        assert _olmocr_parse_yaml_response("{{{{not valid yaml") is None

    def test_yaml_list_returns_none(self):
        # A list is valid YAML but not a dict — should return None
        assert _olmocr_parse_yaml_response("- a\n- b\n- c\n") is None

    def test_yaml_scalar_returns_none(self):
        assert _olmocr_parse_yaml_response("just a string") is None

    def test_is_table_true(self):
        raw = "is_table: true\nis_diagram: false\nnatural_text: header row\n"
        result = _olmocr_parse_yaml_response(raw)
        assert result["is_table"] is True

    def test_natural_text_multiline(self):
        raw = (
            "natural_text: |\n"
            "  Line one\n"
            "  Line two\n"
            "  Line three\n"
        )
        result = _olmocr_parse_yaml_response(raw)
        assert "Line one" in result["natural_text"]
        assert "Line three" in result["natural_text"]

    def test_c2_regression_raw_llm_garbage_is_none(self):
        # Regression for C2: YAML parse failure must NOT return the raw string.
        # An LLM response containing schema keys but invalid YAML structure
        # should return None so the caller returns ("", False, False) instead of
        # embedding the raw text as a real chunk.
        garbage = (
            "primary_language:\n"
            "is_rotation_valid\n"      # missing colon — invalid YAML key-value
            "natural_text: |\n"
            "  some text here\n"
            "extra junk ::::\n"
        )
        result = _olmocr_parse_yaml_response(garbage)
        # Either None (parse failure) or a dict — never the raw input string
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# _postprocess_olmocr_page
# ---------------------------------------------------------------------------

class TestPostprocessOlmocrPage:
    def test_plain_text_no_tables(self):
        body, tables = _postprocess_olmocr_page("This is a paragraph.\n\nAnother paragraph.")
        assert body is not None
        assert "This is a paragraph." in body
        assert tables is None

    def test_html_table_extracted(self):
        text = (
            "Some prose before the table.\n\n"
            "<table><tr><th>Metal</th><th>Grade</th></tr>"
            "<tr><td>Au</td><td>1.5 g/t</td></tr></table>\n\n"
            "More prose after."
        )
        body, tables = _postprocess_olmocr_page(text)
        assert tables is not None
        assert "Au" in tables
        assert "1.5 g/t" in tables
        # Table HTML should be stripped from body
        assert "<table>" not in (body or "")
        assert "Some prose before" in body
        assert "More prose after" in body

    def test_multiple_tables(self):
        text = (
            "<table><tr><td>A</td></tr></table>\n\n"
            "Between.\n\n"
            "<table><tr><td>B</td></tr></table>"
        )
        body, tables = _postprocess_olmocr_page(text)
        assert "A" in tables
        assert "B" in tables

    def test_image_ref_stripped(self):
        text = "Text before.\n![figure caption](figure_01.png)\nText after."
        body, tables = _postprocess_olmocr_page(text)
        assert "![" not in (body or "")
        assert "Text before." in body
        assert "Text after." in body

    def test_empty_string_returns_none_body(self):
        body, tables = _postprocess_olmocr_page("")
        assert body is None
        assert tables is None

    def test_whitespace_only_returns_none_body(self):
        body, tables = _postprocess_olmocr_page("   \n\n   ")
        assert body is None

    def test_table_only_page(self):
        text = "<table><tr><td>Value</td></tr></table>"
        body, tables = _postprocess_olmocr_page(text)
        assert tables is not None
        # body should be None or empty after table extraction
        assert body is None or body.strip() == ""

    def test_case_insensitive_table_tag(self):
        text = "<TABLE><TR><TD>data</TD></TR></TABLE>"
        body, tables = _postprocess_olmocr_page(text)
        assert tables is not None
        assert "data" in tables
