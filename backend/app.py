"""
AI Study Assistant - RAG Backend
Amity University BCA Final Year Project
Supports: .txt, .md, .pdf uploads
PDF requires: pip install pypdf
"""

import os
from dotenv import load_dotenv
load_dotenv()
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
import google.generativeai as genai

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
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

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

def simple_tfidf_score(query: str, chunk: str) -> float:
    q_words      = set(re.findall(r'\w+', query.lower()))
    c_set        = set(re.findall(r'\w+', chunk.lower()))
    if not q_words or not c_set:
        return 0.0
    overlap      = q_words & c_set
    phrase_bonus = 2.0 if query.lower() in chunk.lower() else 0.0
    return round((len(overlap) / len(q_words)) + phrase_bonus, 4)

def retrieve(query: str, top_k: int = 5) -> list:
    chunks = load_json(CHUNKS_FILE, [])
    if not chunks:
        return []
    scored = [(c, simple_tfidf_score(query, c["text"])) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{"chunk": c, "score": s} for c, s in scored[:top_k] if s > 0]

# ── Anthropic API ─────────────────────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

def call_claude(context_chunks: list, question: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        if not context_chunks:
            return "No relevant content found. Please upload study material first."
        snippets = "\n\n".join(f"[{r['chunk']['doc']}] {r['chunk']['text']}" for r in context_chunks[:3])
        return f"Based on your documents:\n\n{snippets}\n\n(Set ANTHROPIC_API_KEY for full AI answers.)"

    context = "\n\n---\n\n".join(
        f"Source: {r['chunk']['doc']}\n{r['chunk']['text']}" for r in context_chunks
    )
    payload = json.dumps({
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system":     "You are a helpful AI study assistant. Answer using ONLY the provided document context. Be clear and cite the source.",
        "messages":   [{"role": "user", "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}"}]
    }).encode()

    req = urllib.request.Request(
        ANTHROPIC_API_URL, data=payload,
        headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["content"][0]["text"]
    except urllib.error.HTTPError as e:
        return f"API error {e.code}: {e.read().decode()}"
    except Exception as e:
        return f"Error: {e}"

# ── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def handle_error(self, request, client_address): pass

    def send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_cors(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length).decode("utf-8", errors="replace")

    def do_OPTIONS(self): self.send_cors()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/documents":
            index = load_json(INDEX_FILE, {})
            self.send_json(200, {"documents": [
                {"name": k, "chunks": v["chunks"], "words": v["words"]} for k, v in index.items()
            ]})
        elif path == "/api/stats":
            chunks = load_json(CHUNKS_FILE, [])
            index  = load_json(INDEX_FILE, {})
            self.send_json(200, {
                "total_chunks": len(chunks),
                "total_docs":   len(index),
                "total_words":  sum(v["words"] for v in index.values()),
                "pdf_support":  PDF_SUPPORTED,
            })
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        body = self.read_body()

        if path == "/api/upload":
            try:
                payload  = json.loads(body)
                doc_name = payload.get("name", "document.txt")
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

                if not text.strip():
                    return self.send_json(400, {"error": "No text could be extracted. PDF may be image-based/scanned."})

                new_chunks = chunk_text(text, doc_name)
                all_chunks = [c for c in load_json(CHUNKS_FILE, []) if c["doc"] != doc_name]
                all_chunks.extend(new_chunks)
                save_json(CHUNKS_FILE, all_chunks)

                index = load_json(INDEX_FILE, {})
                index[doc_name] = {"chunks": len(new_chunks), "words": len(text.split())}
                save_json(INDEX_FILE, index)

                self.send_json(200, {"message": f"Uploaded '{doc_name}' — {len(new_chunks)} chunks indexed.", "chunks": len(new_chunks)})
            except Exception as e:
                self.send_json(500, {"error": str(e)})

        elif path == "/api/ask":
            try:
                payload  = json.loads(body)
                question = payload.get("question", "").strip()
                if not question:
                    return self.send_json(400, {"error": "Empty question"})
                results  = retrieve(question, top_k=5)
                answer   = call_claude(results, question)
                self.send_json(200, {"answer": answer, "sources": list({r["chunk"]["doc"] for r in results}), "chunks_used": len(results)})
            except Exception as e:
                self.send_json(500, {"error": str(e)})

        elif path == "/api/delete":
            try:
                payload  = json.loads(body)
                doc_name = payload.get("name", "")
                save_json(CHUNKS_FILE, [c for c in load_json(CHUNKS_FILE, []) if c["doc"] != doc_name])
                index = load_json(INDEX_FILE, {})
                index.pop(doc_name, None)
                save_json(INDEX_FILE, index)
                self.send_json(200, {"message": f"Deleted '{doc_name}'"})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
        else:
            self.send_json(404, {"error": "Not found"})

if __name__ == "__main__":
    print(f"PDF support: {'YES (pypdf ready)' if PDF_SUPPORTED else 'NO — run: pip install pypdf'}")
    port   = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"AI Study Assistant running on http://localhost:{port}")
    server.serve_forever()