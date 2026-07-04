from core.chunker import chunk_text, _split_sections


def test_short_text_is_single_chunk():
    assert chunk_text("A short policy paragraph.") == ["A short policy paragraph."]

def test_chunks_respect_size_limit():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_no_content_lost():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=400, overlap=40)
    # Every word must appear in at least one chunk
    joined = " ".join(chunks)
    assert all(f"word{i}" in joined for i in range(500))


def test_consecutive_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=400, overlap=100)
    for first, second in zip(chunks, chunks[1:]):
        # The tail of one chunk should share words with the head of the next
        assert set(first.split()) & set(second.split())


def test_numbered_headings_prefix_chunks():
    text = (
        "7.3 Non-Returnable Items\n"
        "Personal hygiene items cannot be returned.\n"
        "7.4 Return Shipping Responsibilities\n"
        "NimbusCart provides a prepaid label for defective items.\n"
    )
    chunks = chunk_text(text)
    assert chunks[0].startswith("7.3 Non-Returnable Items > ")
    assert chunks[1].startswith("7.4 Return Shipping Responsibilities > ")
    assert "prepaid label" in chunks[1]


def test_markdown_headings_prefix_chunks():
    text = "# Refund Policy\nRefunds take 5 days.\n## Exceptions\nGift cards are final."
    chunks = chunk_text(text)
    assert chunks[0].startswith("Refund Policy > ")
    assert chunks[1].startswith("Exceptions > ")


def test_text_before_first_heading_has_no_prefix():
    text = "Introductory paragraph.\n1.1 First Section\nSection body text."
    chunks = chunk_text(text)
    assert chunks[0] == "Introductory paragraph."
    assert chunks[1].startswith("1.1 First Section > ")


def test_bare_numbers_are_not_headings():
    text = "Returns are accepted within\n30 days of delivery in most cases."
    sections = _split_sections(text)
    assert len(sections) == 1
    assert sections[0][0] is None  # no heading detected


def test_long_section_all_chunks_carry_prefix():
    body = " ".join(f"word{i}" for i in range(400))
    text = f"2.1 Long Section\n{body}"
    chunks = chunk_text(text, chunk_size=300, overlap=30)
    assert len(chunks) > 1
    assert all(c.startswith("2.1 Long Section > ") for c in chunks)
