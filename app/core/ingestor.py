import pathlib

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


def load_document(file_path: str) -> str:
    path = pathlib.Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


def load_url(url: str) -> tuple[str, str]:
    """Fetch a web page and return (title, extracted plain text)."""
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; rag-demo-ingestor)"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    # Drop non-content elements before extracting text
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""

    # Rewrite <h1>-<h6> as Markdown headings so the heading-aware chunker can
    # prefix each chunk with its section, same as for uploaded documents.
    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            heading_text = heading.get_text(strip=True)
            if heading_text:
                heading.string = f"\n{'#' * level} {heading_text}\n"
    lines = (line.strip() for line in soup.get_text("\n").splitlines())
    text = "\n".join(line for line in lines if line)
    return title, text
