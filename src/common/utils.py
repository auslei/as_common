"""General utility functions."""

import re
import markdown2
from typing import Optional

def sanitize_project_name(name: str) -> str:
    """Sanitize a name for use in file system paths and collection identifiers."""
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^\w\-一-鿿]+", "", name)
    return name

def render_markdown_to_html(markdown_text: str, title: str = "Documentation") -> str:
    """Render markdown text into a full HTML document with basic styling."""
    html_snippet = markdown2.markdown(
        markdown_text,
        extras=["tables", "fenced-code-blocks", "code-friendly", "task_list", "header-ids"]
    )

    css = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        line-height: 1.6;
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem;
        color: #24292e;
        background-color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6 {
        margin-top: 24px;
        margin-bottom: 16px;
        font-weight: 600;
        line-height: 1.25;
        border-bottom: 1px solid #eaecef;
        padding-bottom: 0.3em;
    }
    h1 { font-size: 2em; }
    h2 { font-size: 1.5em; }
    pre {
        background-color: #f6f8fa;
        padding: 16px;
        border-radius: 6px;
        overflow: auto;
        font-size: 85%;
        line-height: 1.45;
    }
    code {
        background-color: rgba(27,31,35,0.05);
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 85%;
    }
    pre code {
        background-color: transparent;
        padding: 0;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 16px;
    }
    th, td {
        border: 1px solid #dfe2e5;
        padding: 6px 13px;
    }
    tr:nth-child(even) {
        background-color: #f6f8fa;
    }
    blockquote {
        padding: 0 1em;
        color: #6a737d;
        border-left: 0.25em solid #dfe2e5;
        margin: 0 0 16px 0;
    }
    ul, ol {
        padding-left: 2em;
    }
    li + li {
        margin-top: 0.25em;
    }
    img {
        max-width: 100%;
    }
    hr {
        height: 0.25em;
        padding: 0;
        margin: 24px 0;
        background-color: #e1e4e8;
        border: 0;
    }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="markdown-body">
{html_snippet}
    </div>
</body>
</html>
"""
