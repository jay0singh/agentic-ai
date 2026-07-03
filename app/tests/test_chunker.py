from core.chunker import chunk_text


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
