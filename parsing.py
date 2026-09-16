"""
Parses Confluence's storage-format XHTML into structured chunks that keep
headings, tables, and code blocks intact - instead of flattening everything
to prose the way the original repos did (which is exactly the content most
likely to be asked about: config values, comparison tables, code samples).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag


def _token_count(text: str) -> int:
    """Approximate token count via the standard ~4-chars-per-token heuristic
    (OpenAI's own rule of thumb for English text). Deliberately not using
    tiktoken here: it needs a network call on first use to fetch its BPE file,
    which would make chunking - a pure offline operation - depend on internet
    access. This is off by a few percent at worst, which doesn't matter for
    chunk-sizing decisions."""
    return max(1, len(text) // 4)


@dataclass
class Section:
    """One heading-delimited section of a page, with breadcrumb context."""

    breadcrumb: str  # e.g. "GPT-5 > Specs > Context Window"
    heading: str
    content: str  # markdown-ish plain text, tables preserved as markdown tables


@dataclass
class Chunk:
    page_id: str
    title: str
    version: int
    url: str
    breadcrumb: str
    section: str
    content: str
    chunk_index: int
    token_count: int = field(init=False)

    def __post_init__(self):
        self.token_count = _token_count(self.content)

    @property
    def id(self) -> str:
        return f"{self.page_id}::{self.version}::{self.chunk_index}"


def _table_to_markdown(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        # pad/truncate rows that don't match header width, rather than crash
        row = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _code_macro_to_fenced_block(macro: Tag) -> str:
    lang_param = macro.find("ac:parameter", {"ac:name": "language"})
    lang = lang_param.get_text(strip=True) if lang_param else ""
    body = macro.find("ac:plain-text-body")
    code = body.get_text() if body else macro.get_text()
    return f"```{lang}\n{code.strip()}\n```"


def _walk_body_to_blocks(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """
    Walk the top-level body and emit a flat list of (kind, text) blocks:
    kind in {"h1","h2","h3","h4","p","table","code"}. Tables and code
    macros are converted to Markdown-safe text instead of being flattened.
    """
    blocks: list[tuple[str, str]] = []

    def handle(node: Tag):
        if not isinstance(node, Tag):
            return
        name = node.name

        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            blocks.append((name, node.get_text(" ", strip=True)))
            return

        if name == "table":
            md = _table_to_markdown(node)
            if md:
                blocks.append(("table", md))
            return

        if name == "ac:structured-macro" and node.get("ac:name") == "code":
            blocks.append(("code", _code_macro_to_fenced_block(node)))
            return

        # Don't descend into elements we've already fully handled above,
        # or we'd double-emit their text as a paragraph too.
        if name in ("table",):
            return

        # Leaf-ish text content (paragraphs, list items, etc.)
        if name in ("p", "li"):
            text = node.get_text(" ", strip=True)
            if text:
                blocks.append(("p", text))
            return

        # Recurse into containers (div, body, ac:rich-text-body, ul, ol, etc.)
        for child in node.find_all(recursive=False):
            handle(child)

    handle(soup)
    return blocks


def parse_storage_html(html: str) -> list[Section]:
    """Turn Confluence storage-format HTML into heading-delimited sections."""
    soup = BeautifulSoup(html, "html.parser")
    blocks = _walk_body_to_blocks(soup)

    sections: list[Section] = []
    heading_stack: list[str] = []  # current breadcrumb path
    current_heading = "Overview"
    current_lines: list[str] = []

    def flush():
        if current_lines:
            sections.append(
                Section(
                    breadcrumb=" > ".join(heading_stack) if heading_stack else current_heading,
                    heading=current_heading,
                    content="\n\n".join(current_lines).strip(),
                )
            )

    for kind, text in blocks:
        if kind.startswith("h") and kind[1:].isdigit():
            flush()
            level = int(kind[1:])
            # trim the breadcrumb stack back to this heading level
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(text)
            current_heading = text
            current_lines = []
        else:
            current_lines.append(text)

    flush()
    return [s for s in sections if s.content]


_WORD_SPLIT = re.compile(r"\s+")


def _is_protected_block(para: str) -> bool:
    """Table markdown and fenced code blocks must never be split mid-line -
    everything else (plain prose) can fall back to a word-level split."""
    return para.startswith("|") or para.startswith("```")


def _sliding_window_join(units: list[str], sep: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack `units` into pieces joined by `sep`, each <= max_tokens,
    carrying a token-bounded overlap tail of whole units into the next piece.
    Token counts are measured on the actual joined text rather than summed
    per-unit - summing per-unit undercounts separator overhead once units
    are short and numerous (e.g. word-level splitting), which would let
    pieces silently exceed max_tokens."""
    pieces: list[str] = []
    current: list[str] = []

    for unit in units:
        candidate = current + [unit]
        if current and _token_count(sep.join(candidate)) > max_tokens:
            pieces.append(sep.join(current))
            overlap: list[str] = []
            for u in reversed(current):
                candidate_overlap = [u] + overlap
                if _token_count(sep.join(candidate_overlap)) > overlap_tokens:
                    break
                overlap = candidate_overlap
            current = overlap
        current.append(unit)

    if current:
        pieces.append(sep.join(current))
    return pieces


def _split_paragraph_by_words(para: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Word-level fallback split, used when a single paragraph (no blank-line
    breaks of its own) alone exceeds max_tokens - e.g. a wall-of-text block."""
    return _sliding_window_join(_WORD_SPLIT.split(para.strip()), " ", max_tokens, overlap_tokens)


def _split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Token-aware sliding window split, used only when a single section
    exceeds max_tokens on its own. Never splits a table row/fence mid-line -
    those fall through whole; oversized prose paragraphs are word-split so a
    single wall-of-text paragraph doesn't produce one giant chunk."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    expanded: list[str] = []
    for para in paragraphs:
        if _token_count(para) > max_tokens and not _is_protected_block(para):
            expanded.extend(_split_paragraph_by_words(para, max_tokens, overlap_tokens))
        else:
            expanded.append(para)
    return _sliding_window_join(expanded, "\n\n", max_tokens, overlap_tokens)


def chunk_page(
    *,
    page_id: str,
    title: str,
    version: int,
    url: str,
    html: str,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Section-aware chunking: split on heading boundaries first, then
    token-aware sliding window *within* a section, so a chunk never straddles
    two unrelated headings and tables never get split mid-row."""
    sections = parse_storage_html(html)
    chunks: list[Chunk] = []
    idx = 0

    for section in sections:
        if _token_count(section.content) <= max_tokens:
            pieces = [section.content]
        else:
            pieces = _split_by_tokens(section.content, max_tokens, overlap_tokens)

        for piece in pieces:
            full_breadcrumb = (
                section.breadcrumb
                if section.breadcrumb.startswith(title)
                else f"{title} > {section.breadcrumb}"
            )
            chunks.append(
                Chunk(
                    page_id=page_id,
                    title=title,
                    version=version,
                    url=url,
                    breadcrumb=full_breadcrumb,
                    section=section.heading,
                    content=piece,
                    chunk_index=idx,
                )
            )
            idx += 1

    return chunks
