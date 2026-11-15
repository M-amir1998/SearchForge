"""Format-specific parsing helpers (Markdown front matter/title, HTML to text)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

import yaml

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


@dataclass(slots=True)
class ParsedMarkdown:
    body: str
    front_matter: dict[str, Any] = field(default_factory=dict)
    title: str | None = None


def parse_markdown(text: str) -> ParsedMarkdown:
    """Split YAML front matter (safe_load only) and detect the first H1 as title."""
    front: dict[str, Any] = {}
    body = text
    match = _FRONT_MATTER.match(text)
    if match:
        try:
            loaded = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            loaded = None
        if isinstance(loaded, dict):
            front = {str(k): v for k, v in loaded.items()}
            body = text[match.end() :]
    title = front.get("title") if isinstance(front.get("title"), str) else None
    if title is None:
        for m in _ATX_HEADING.finditer(body):
            if len(m.group(1)) == 1:
                title = m.group(2).strip()
                break
    return ParsedMarkdown(body=body, front_matter=front, title=title)


def markdown_headings(text: str) -> list[tuple[int, int, str]]:
    """Return ``(offset, level, heading_text)`` for ATX headings outside code fences."""
    out: list[tuple[int, int, str]] = []
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence:
            m = _ATX_HEADING.match(line.rstrip("\n"))
            if m:
                out.append((offset, len(m.group(1)), m.group(2).strip()))
        offset += len(line)
    return out


class _TextExtractor(HTMLParser):
    _SKIP = frozenset({"script", "style", "noscript", "template", "svg", "iframe", "object"})
    _BLOCK = frozenset(
        {
            "p",
            "div",
            "br",
            "li",
            "tr",
            "section",
            "article",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "pre",
            "blockquote",
            "table",
            "ul",
            "ol",
            "header",
            "footer",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._heading: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
            self._heading = tag
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag == self._heading:
            self.parts.append("\n\n")
            self._heading = None
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str | None]:
    """Extract visible text from untrusted HTML. Scripts/styles are dropped, never executed.

    Headings are rendered as Markdown ATX headings so the section chunker still works.
    """
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    title = " ".join("".join(parser.title_parts).split()) or None
    return text, title
