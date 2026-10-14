import importlib
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()
# ✓/✗ can't be printed by a Windows console using cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PASS = "✓"
FAIL = "✗"
errors = []
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    errors.append("GOOGLE_API_KEY not set in .env file")
checks = [
    # (the old check imported the `langchain` package, which this project
    # doesn't install, so it always reported "MISSING")
    ("LangChain text splitters", "langchain_text_splitters"),
    ("LangChain Community",     "langchain_community"),
    ("LangChain Google GenAI",  "langchain_google_genai"),
    ("FAISS",                   "faiss"),
    ("BM25",                    "rank_bm25"),
    ("PyPDF",                   "pypdf"),
    ("FastAPI",                 "fastapi"),
    ("python-dotenv",           "dotenv"),
]
print("\nRAG Chatbot — Environment Check\n" + "-" * 34)
for name, imp in checks:
    try:
        importlib.import_module(imp)
        print(f"  {PASS}  {name}")
    except ImportError:
        print(f"  {FAIL}  {name}  [MISSING]")
        errors.append(f"{name} not installed")
print("-" * 34)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    response = llm.invoke("Reply with exactly: SETUP SUCCESSFUL")
    assert "SUCCESSFUL" in response.content
    print(f"  {PASS}  Gemini API connection")
except Exception as e:
    msg = str(e)
    reason = (
        "Rate limit — retry in 1 minute" if "RESOURCE_EXHAUSTED" in msg else
        "Model unavailable on this account" if "404" in msg else
        "Invalid API key" if "401" in msg else
        msg[:80]
    )
    print(f"  {FAIL}  Gemini API  [{reason}]")
    errors.append(reason)
print("-" * 34)
if errors:
    print(f"\n  {len(errors)} issue(s) found:")
    for e in errors:
        print(f"    • {e}")
    sys.exit(1)
else:
    print("\n  All checks passed. Ready for Next Phase.\n")