import os
import sys

# Dummy credentials so modules that build clients at import time work in CI.
# load_dotenv() never overrides existing env vars, so these also isolate local
# test runs from the developer's real .env where it matters.
os.environ.setdefault("GROQ_API_KEY", "gsk-test-dummy")
os.environ.setdefault("GOOGLE_API_KEY", "test-dummy")
os.environ.setdefault("DB_NAME", "testdb")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5433")
os.environ.setdefault("DB_TABLE", "documents")

# Make the app package root (app/) importable regardless of pytest's cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
