import pathlib
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader


def load_document(file_path: str) -> str:
    path = pathlib.Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)