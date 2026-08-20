"""
AI Study Assistant - RAG Backend
Amity University BCA Final Year Project
Author: Alex

Supports: .txt, .md, .pdf uploads
Requires: pip install pypdf python-dotenv
"""

import os
import io
import json
import base64
import hashlib
import re
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import urllib.request
import urllib.error

from dotenv import load_dotenv
load_dotenv()

# ── PDF support ───────────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    PDF_SUPPORTED = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PDF_SUPPORTED = True
    except ImportError:
        PDF_SUPPORTED = False

# ── Storage ───────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CHUNKS_FILE = DATA_DIR / "chunks.json"
INDEX_FILE  = DATA_DIR / "index.json"

def load_json(path, default):
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── PDF Extraction ────────────────────────────────────────────────────────────
def extract_pdf_text(b64_data: str) -> str:
    raw    = base64.b64decode(b64_data)
    buf    = io.BytesIO(raw)
    reader = PdfReader(buf)
    pages  = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            pages.append(f"[Page {i+1}]\n{t.strip()}")
    return "\n\n".join(pages)

# ── Text Processing ───────────────────────────────────────────────────────────
def chunk_text(text: str, doc_name: str, chunk_size: int = 400, overlap: int = 80) -> list:
    words  = text.split()
    chunks = []
    start  = 0
    idx    = 0
    while start < len(words):
        end  = min(start + chunk_size, len(words))
        body = " ".join(words[start:end])
        chunks.append({
            "id":      hashlib.md5(f"{doc_name}-{idx}".encode()).hexdigest()[:8],
            "doc":     doc_name,
            "text":    body,
            "preview": body[:120] + ("..." if len(body) > 120 else ""),
        })
        start += chunk_size - overlap
        idx   += 1
    return chunks

def tfidf_score(query: str, chunk: str) -> float:
    q_words      = set(re.findall(r'\w+', query.lower()))
    c_words      = set(re.findall(r'\w+', chunk.lower()))
    if not q_words or not c_words:
        return 0.0
    overlap      = q_words & c_words
    phrase_bonus = 2.0 if query.lower() in chunk.lower() else 0.0
    return round((len(overlap) / len(q_words)) + phrase_bonus, 4)

def retrieve(query: str, top_k: int = 5) -> list:
    chunks = load_json(CHUNKS_FILE, [])
    if not chunks:
        return []
    scored = [(c, tfidf_score(query, c["text"])) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"chunk": c, "score": s} for c, s in scored[:top_k] if s > 0]

# ── Claude API ────────────────────────────────────────────────────────────────
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-opus-4-5"   # best model for answers
# CLAUDE_MODEL = "claude-haiku-4-5"   # cheaper model

def call_claude(context_chunks: list, question: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    # ── No API key ─────────────────────────────────────────────────────────
    if not api_key:
        if not context_chunks:
            return "⚠️ No study material uploaded yet. Please upload a document first."
        snippets = "\n\n".join(
            f"📄 [{r['chunk']['doc']}]\n{r['chunk']['text']}" for r in context_chunks[:3]
        )
        return f"**Retrieved from your documents:**\n\n{snippets}\n\n---\n⚠️ Add ANTHROPIC_API_KEY to .env for full AI answers."

    # ── No relevant chunks found ────────────────────────────────────────────
    if not context_chunks:
        return "❌ No relevant content found in your uploaded documents. Try rephrasing your question."

    # ── Build context ───────────────────────────────────────────────────────
    context = "\n\n---\n\n".join(
        f"Source: {r['chunk']['doc']} (relevance: {r['score']})\n{r['chunk']['text']}"
        for r in context_chunks
    )

    system_prompt = """You are an expert AI study assistant helping a university student.

Your rules:
- Answer ONLY using the provided document context
- Be clear, concise and educational
- Always mention which source document the answer comes from
- If the answer is NOT in the context, say: "This topic is not covered in your uploaded documents."
- Format your answer nicely with short paragraphs
- Do not make up information"""

    user_prompt = f"""Here is the relevant content from the student's study material:

{context}

---

Student's question: {question}

Please answer clearly and cite the source document."""

    payload = json.dumps({
        "model":      CLAUDE_MODEL,
        "max_tokens": 1024,
        "system":     system_prompt,
        "messages":   [{"role": "user", "content": user_prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        CLAUDE_API_URL,
        data    = payload,
        headers = {
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        # friendly error messages
        if "credit" in err.lower() or "402" in str(e.code) or "400" in str(e.code):
            return "❌ Insufficient API credits. Please top up at console.anthropic.com → Billing."
        if "401" in str(e.code):
            return "❌ Invalid API key. Check your ANTHROPIC_API_KEY in .env"
        return f"❌ API error {e.code}: {err}"
    except Exception as e:
        return f"❌ Connection error: {e}"

# ── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args): pass
    def handle_error(self, request, client_address): pass

    def send_json(self, status: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def read_body(self) -> str:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length).decode("utf-8", errors="replace")

    # ── GET ───────────────────────────────────────────────────────────────────
    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/documents":
            index = load_json(INDEX_FILE, {})
            self.send_json(200, {"documents": [
                {"name": k, "chunks": v["chunks"], "words": v["words"]}
                for k, v in index.items()
            ]})

        elif path == "/api/stats":
            chunks = load_json(CHUNKS_FILE, [])
            index  = load_json(INDEX_FILE, {})
            has_key = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
            self.send_json(200, {
                "total_chunks": len(chunks),
                "total_docs":   len(index),
                "total_words":  sum(v["words"] for v in index.values()),
                "pdf_support":  PDF_SUPPORTED,
                "api_ready":    has_key,
            })

        elif path == "/api/health":
            self.send_json(200, {"status": "ok", "pdf": PDF_SUPPORTED})

        else:
            self.send_json(404, {"error": "Not found"})

    # ── POST ──────────────────────────────────────────────────────────────────
    def do_POST(self):
        path = urlparse(self.path).path
        body = self.read_body()

        # ── Upload ────────────────────────────────────────────────────────────
        if path == "/api/upload":
            try:
                payload  = json.loads(body)
                doc_name = payload.get("name", "document.txt").strip()
                is_pdf   = payload.get("is_pdf", False)

                if is_pdf:
                    if not PDF_SUPPORTED:
                        return self.send_json(400, {"error": "PDF not supported. Run: pip install pypdf"})
                    b64 = payload.get("data", "")
                    if not b64:
                        return self.send_json(400, {"error": "No PDF data received"})
                    try:
                        text = extract_pdf_text(b64)
                    except Exception as e:
                        return self.send_json(500, {"error": f"PDF extraction failed: {e}"})
                else:
                    text = payload.get("text", "")

                text = text.strip()
                if not text:
                    return self.send_json(400, {
                        "error": "No text extracted. PDF may be scanned/image-based — try copy-pasting the text instead."
                    })

                new_chunks = chunk_text(text, doc_name)
                all_chunks = [c for c in load_json(CHUNKS_FILE, []) if c["doc"] != doc_name]
                all_chunks.extend(new_chunks)
                save_json(CHUNKS_FILE, all_chunks)

                index = load_json(INDEX_FILE, {})
                index[doc_name] = {"chunks": len(new_chunks), "words": len(text.split())}
                save_json(INDEX_FILE, index)

                self.send_json(200, {
                    "message": f"✅ '{doc_name}' indexed successfully — {len(new_chunks)} chunks, {len(text.split())} words.",
                    "chunks":  len(new_chunks),
                    "words":   len(text.split()),
                })
            except json.JSONDecodeError:
                self.send_json(400, {"error": "Invalid JSON in request body"})
            except Exception as e:
                self.send_json(500, {"error": str(e)})

        # ── Ask ───────────────────────────────────────────────────────────────
        elif path == "/api/ask":
            try:
                payload  = json.loads(body)
                question = payload.get("question", "").strip()
                if not question:
                    return self.send_json(400, {"error": "Please enter a question."})

                results = retrieve(question, top_k=5)
                answer  = call_claude(results, question)

                self.send_json(200, {
                    "answer":      answer,
                    "sources":     list({r["chunk"]["doc"] for r in results}),
                    "chunks_used": len(results),
                    "top_score":   results[0]["score"] if results else 0,
                })
            except Exception as e:
                self.send_json(500, {"error": str(e)})

        # ── Delete ────────────────────────────────────────────────────────────
        elif path == "/api/delete":
            try:
                payload  = json.loads(body)
                doc_name = payload.get("name", "")
                if not doc_name:
                    return self.send_json(400, {"error": "No document name provided"})

                save_json(CHUNKS_FILE, [
                    c for c in load_json(CHUNKS_FILE, []) if c["doc"] != doc_name
                ])
                index = load_json(INDEX_FILE, {})
                index.pop(doc_name, None)
                save_json(INDEX_FILE, index)
                self.send_json(200, {"message": f"🗑️ '{doc_name}' deleted successfully."})
            except Exception as e:
                self.send_json(500, {"error": str(e)})

        else:
            self.send_json(404, {"error": "Not found"})

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    print("=" * 50)
    print("   AI Study Assistant - RAG Backend")
    print("=" * 50)
    print(f"PDF support : {'YES' if PDF_SUPPORTED else 'NO  → pip install pypdf'}")
    print(f"Claude API  : {'READY ✓' if api_key else 'NOT SET ✗ → add to .env'}")
    print(f"Model       : {CLAUDE_MODEL}")
    port = int(os.environ.get("PORT", 8000))
    print(f"Server      : http://localhost:{port}")
    print("=" * 50)
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()