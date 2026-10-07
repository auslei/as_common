import pytest
from src.common.utils import sanitize_project_name, render_markdown_to_html


def test_sanitize_project_name():
    assert sanitize_project_name("My Project") == "My_Project"
    assert sanitize_project_name("Project!@#$%^&*()") == "Project"
    assert sanitize_project_name("Project-123_abc") == "Project-123_abc"
    assert sanitize_project_name("项目名称") == "项目名称"
    assert sanitize_project_name("  Trim  And  Space  ") == "_Trim_And_Space_"


def test_render_markdown_to_html():
    md = "# Hello\nThis is a test."
    html = render_markdown_to_html(md, title="Test Title")
    assert "<title>Test Title</title>" in html
    assert '<h1 id="hello">Hello</h1>' in html
    assert "<p>This is a test.</p>" in html
    assert "markdown-body" in html


def test_render_markdown_to_html_default_title():
    html = render_markdown_to_html("# Hi")
    assert "<title>Documentation</title>" in html
