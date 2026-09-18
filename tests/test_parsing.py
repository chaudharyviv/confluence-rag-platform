"""
Unit tests for parsing.py's chunking logic - specifically the invariants
that matter for retrieval quality and that an eval run wouldn't necessarily
catch (a broken table or a chunk straddling two headings might not shift
the golden set's answers, but it's still wrong): tables and code fences
are never split mid-block, chunks respect heading boundaries, and the
token-aware sliding window actually stays under its cap.
"""
from parsing import chunk_page, parse_storage_html, _token_count


def _table_html(rows: int) -> str:
    header = "<tr><th>Model</th><th>Context</th><th>License</th></tr>"
    body = "".join(
        f"<tr><td>Model-{i}</td><td>{i}k tokens</td><td>Open</td></tr>" for i in range(rows)
    )
    return f"<table>{header}{body}</table>"


def _code_macro(lang: str, code: str) -> str:
    return (
        '<ac:structured-macro ac:name="code">'
        f'<ac:parameter ac:name="language">{lang}</ac:parameter>'
        f"<ac:plain-text-body><![CDATA[{code}]]></ac:plain-text-body>"
        "</ac:structured-macro>"
    )


def test_table_rendered_as_markdown_and_never_split():
    html = f"<h1>Specs</h1><p>Intro text.</p>{_table_html(3)}"
    chunks = chunk_page(page_id="p1", title="Specs Page", version=1, url="http://x", html=html)
    joined = "\n".join(c.content for c in chunks)
    assert "| Model | Context | License |" in joined
    assert "| --- | --- | --- |" in joined
    # every table row present, and each row lives inside a single chunk
    for i in range(3):
        row = f"Model-{i}"
        matching = [c for c in chunks if row in c.content]
        assert len(matching) == 1
        assert "| ---" in matching[0].content  # the row's chunk kept its header


def test_code_fence_preserved_and_not_split_mid_block():
    code = "def f():\n    return 1\n" * 5
    html = f"<h1>Snippet</h1>{_code_macro('python', code)}"
    chunks = chunk_page(page_id="p1", title="Snippet Page", version=1, url="http://x", html=html)
    joined = "\n".join(c.content for c in chunks)
    assert "```python" in joined
    assert code.strip() in joined
    fenced = [c for c in chunks if "```python" in c.content]
    assert len(fenced) == 1
    assert fenced[0].content.count("```") == 2  # opening + closing fence, same chunk


def test_sections_split_on_heading_boundaries():
    html = (
        "<h1>Alpha</h1><p>Alpha content.</p>"
        "<h1>Beta</h1><p>Beta content.</p>"
    )
    sections = parse_storage_html(html)
    assert [s.heading for s in sections] == ["Alpha", "Beta"]
    assert "Alpha content" in sections[0].content
    assert "Beta content" not in sections[0].content


def test_breadcrumb_reflects_nested_headings():
    html = "<h1>GPT-5</h1><h2>Specs</h2><h3>Context Window</h3><p>1M tokens.</p>"
    chunks = chunk_page(page_id="p1", title="GPT-5", version=1, url="http://x", html=html)
    assert chunks[-1].breadcrumb == "GPT-5 > Specs > Context Window"


def test_long_section_is_split_under_token_cap_with_overlap():
    # ~1000 words of prose, well over a 50-token cap
    prose = " ".join(f"word{i}" for i in range(1000))
    html = f"<h1>Wall of text</h1><p>{prose}</p>"
    chunks = chunk_page(
        page_id="p1", title="Doc", version=1, url="http://x", html=html,
        max_tokens=50, overlap_tokens=10,
    )
    assert len(chunks) > 1
    for c in chunks:
        assert _token_count(c.content) <= 50 + 5  # small slack for separator overhead
    # consecutive chunks share some overlap content (last words of one appear in the next)
    tail_words = chunks[0].content.split()[-3:]
    assert any(w in chunks[1].content for w in tail_words)


def test_chunk_index_and_id_increment_within_page():
    html = "<h1>A</h1><p>one</p><h1>B</h1><p>two</p>"
    chunks = chunk_page(page_id="p42", title="Doc", version=3, url="http://x", html=html)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"p42::3::{c.chunk_index}" for c in chunks)
