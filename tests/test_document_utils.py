import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure document_utils can be imported even if pypandoc is not installed
if "pypandoc" not in sys.modules:
    sys.modules["pypandoc"] = MagicMock()

from src.common.document_utils import _fix_markdown_lists_and_tables, md_to_pdf


class TestFixMarkdownListsAndTables:

    def test_plain_text_unchanged(self):
        content = "Just plain text\nnothing special here"
        assert _fix_markdown_lists_and_tables(content) == content

    def test_adds_blank_line_before_dash_list_after_text(self):
        content = "Intro text\n- item1\n- item2"
        result = _fix_markdown_lists_and_tables(content)
        assert "\n\n- item1" in result

    def test_adds_blank_line_before_star_list_after_text(self):
        content = "Header\n* first item"
        result = _fix_markdown_lists_and_tables(content)
        assert "\n\n* first item" in result

    def test_adds_blank_line_before_numbered_list_after_text(self):
        content = "Header\n1. first\n2. second"
        result = _fix_markdown_lists_and_tables(content)
        lines = result.split("\n")
        list_idx = next(i for i, l in enumerate(lines) if l.startswith("1."))
        assert lines[list_idx - 1] == ""

    def test_no_extra_blank_when_already_present(self):
        content = "Some text\n\n- item1"
        result = _fix_markdown_lists_and_tables(content)
        assert "\n\n\n" not in result

    def test_no_blank_added_between_consecutive_list_items(self):
        content = "- item1\n- item2\n- item3"
        result = _fix_markdown_lists_and_tables(content)
        assert result.count("\n\n") == 0

    def test_adds_blank_line_before_table_after_text(self):
        content = "Header\n| col1 | col2 |\n| --- | --- |"
        result = _fix_markdown_lists_and_tables(content)
        assert "\n\n| col1 |" in result

    def test_no_blank_line_added_to_first_line_list(self):
        content = "- item1\n- item2"
        result = _fix_markdown_lists_and_tables(content)
        assert not result.startswith("\n")

    def test_no_blank_line_added_to_first_line_table(self):
        content = "| col | val |\n| --- | --- |"
        result = _fix_markdown_lists_and_tables(content)
        assert not result.startswith("\n")

    def test_blockquote_not_treated_as_table(self):
        content = "Text\n> | not a table |"
        result = _fix_markdown_lists_and_tables(content)
        # No blank line should be added before a blockquote with pipe
        assert "\n\n>" not in result

    def test_preserves_existing_content(self):
        content = "# Title\n\nSome paragraph.\n\n- list item\n"
        result = _fix_markdown_lists_and_tables(content)
        assert "# Title" in result
        assert "Some paragraph." in result
        assert "- list item" in result


class TestMdToPdf:

    def test_calls_pypandoc_convert_text(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nThis is a test.")
        output_pdf = str(tmp_path / "output.pdf")

        with patch("src.common.document_utils.pypandoc.convert_text") as mock_convert:
            md_to_pdf(str(md_file), output_pdf)
            assert mock_convert.called
            call_kwargs = mock_convert.call_args.kwargs
            assert call_kwargs["to"] == "pdf"
            assert call_kwargs["outputfile"] == output_pdf

    def test_handles_missing_pandoc_gracefully(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello")

        with patch("src.common.document_utils.pypandoc.convert_text",
                   side_effect=OSError("pandoc not found")):
            md_to_pdf(str(md_file), str(tmp_path / "out.pdf"))  # must not raise

    def test_retries_with_fallback_font_on_font_error(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# CJK 中文")

        call_count = 0

        def fail_first_font(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fontspec error: Noto Sans CJK SC not found")

        with patch("src.common.document_utils.pypandoc.convert_text",
                   side_effect=fail_first_font):
            md_to_pdf(str(md_file), str(tmp_path / "out.pdf"))
            assert call_count >= 2

    def test_reads_markdown_file_content(self, tmp_path):
        md_file = tmp_path / "content.md"
        md_file.write_text("# My Report\n\nSome content here.")

        captured = {}

        def capture_call(source, *args, **kwargs):
            captured["source"] = source

        with patch("src.common.document_utils.pypandoc.convert_text",
                   side_effect=capture_call):
            md_to_pdf(str(md_file), str(tmp_path / "out.pdf"))
            assert "My Report" in captured.get("source", "")
