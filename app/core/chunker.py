import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Heading heuristics, kept conservative to avoid false positives in body text:
# - Markdown headings: "# Title" ... "###### Title"
# - Numbered sections with at least one dot: "7.4 Return Shipping" (a bare
#   number like "30 days later" must NOT match)
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)+\.?\s+[A-Z][^\n]{0,80}$")


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split text into (heading, body) sections at heading lines.
    Content before the first heading gets heading=None."""
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append((heading, body))
        buffer.clear()

    for line in text.splitlines():
        stripped = line.strip()
        md = _MD_HEADING.match(stripped)
        if md or _NUMBERED_HEADING.match(stripped):
            flush()
            heading = md.group(1).strip() if md else stripped
        else:
            buffer.append(line)
    flush()
    return sections


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Heading-aware chunking: split the document into sections at headings,
    chunk each section, and prefix every chunk with its section heading so
    retrieval and citations carry the document structure."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )

    chunks: list[str] = []
    for heading, body in _split_sections(text):
        prefix = f"{heading} > " if heading else ""
        for piece in splitter.split_text(body):
            chunks.append(prefix + piece)
    return chunks
