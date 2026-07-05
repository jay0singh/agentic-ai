import pytest

import core.ingestor as ingestor


def test_load_txt(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Plain text policy notes.", encoding="utf-8")
    assert ingestor.load_document(str(f)) == "Plain text policy notes."


def test_load_md(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text("# Heading\n\nSome markdown body.", encoding="utf-8")
    assert "Some markdown body." in ingestor.load_document(str(f))


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError):
        ingestor.load_document(str(f))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ingestor.load_document("does-not-exist.txt")


def _fake_url_response(monkeypatch, html):
    class FakeResponse:
        text = html
        def raise_for_status(self):
            pass

    monkeypatch.setattr(ingestor.requests, "get", lambda *a, **k: FakeResponse())


def test_load_url_extracts_title_and_text(monkeypatch):
    _fake_url_response(monkeypatch, """
    <html><head><title> Sample Page </title><script>ignored()</script></head>
    <body>
      <nav>menu junk</nav>
      <h1>Welcome</h1>
      <p>This is the real content.</p>
      <footer>footer junk</footer>
    </body></html>
    """)

    title, text = ingestor.load_url("https://example.com/page")
    assert title == "Sample Page"
    assert "This is the real content." in text
    assert "menu junk" not in text
    assert "footer junk" not in text
    assert "ignored()" not in text


def test_load_url_converts_html_headings_to_markdown(monkeypatch):
    _fake_url_response(monkeypatch, """
    <html><body>
      <h1>Design principles</h1>
      <p>We follow five principles.</p>
      <h2>Secure by default</h2>
      <p>Auth is built in.</p>
    </body></html>
    """)

    _, text = ingestor.load_url("https://example.com/page")
    assert "# Design principles" in text
    assert "## Secure by default" in text
    # And the chunker turns those into section-prefixed chunks:
    from core.chunker import chunk_text
    chunks = chunk_text(text)
    assert any(c.startswith("Secure by default > ") for c in chunks)
