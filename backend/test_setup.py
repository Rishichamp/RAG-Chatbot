import os
import sys
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()
PASS = "✓"
FAIL = "✗"
errors = []
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    errors.append("GOOGLE_API_KEY not set in .env file")
checks = [
    ("LangChain",               "from langchain.text_splitter import RecursiveCharacterTextSplitter"),
    ("LangChain Google GenAI",  "from langchain_google_genai import ChatGoogleGenerativeAI"),
    ("FAISS",                   "import faiss"),
    ("BM25",                    "from rank_bm25 import BM25Okapi"),
    ("PyPDF",                   "import pypdf"),
    ("FastAPI",                 "import fastapi"),
    ("python-dotenv",           "import dotenv"),
]
print("\nRAG Chatbot — Environment Check\n" + "-" * 34)
for name, imp in checks:
    try:
        exec(imp)
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