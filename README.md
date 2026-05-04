# AI Study Assistant — RAG-Based Application
**Amity University Online · BCA Final Year Project**

A full-stack AI-powered study assistant that lets students upload lecture notes and ask natural-language questions, answered using Retrieval-Augmented Generation (RAG).

---

## Architecture

```
User → Frontend (HTML/JS) → Python REST API → RAG Pipeline → Claude AI → Answer
                                            ↕
                                   Vector Store (JSON chunks)
```

## Features
- Upload .txt / .md files or paste notes directly
- Automatic chunking with 80-word overlap for better retrieval
- TF-IDF keyword scoring for offline retrieval (no external DB needed)
- Claude AI integration for intelligent, cited answers
- Real-time document stats (chunks, words, documents)
- Delete individual documents from the index

## Setup & Run

### Prerequisites
- Python 3.10+
- (Optional) Anthropic API key for AI-powered answers

### Start Backend
```bash
# Optional: set your API key
export ANTHROPIC_API_KEY=your_key_here

cd backend
python app.py
# Runs on http://localhost:8000
```

### Open Frontend
Open `frontend/index.html` in your browser — or serve it:
```bash
cd frontend
python -m http.server 3000
# Visit http://localhost:3000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents` | List all indexed documents |
| GET | `/api/stats` | Total docs, chunks, words |
| POST | `/api/upload` | Upload and index a document |
| POST | `/api/ask` | Ask a question (RAG + AI) |
| POST | `/api/delete` | Remove a document |

## Tech Stack
- **Backend**: Python 3, stdlib only (no pip installs needed)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **AI**: Anthropic Claude API (claude-sonnet-4-20250514)
- **Storage**: JSON flat files (chunks.json, index.json)
- **Retrieval**: TF-IDF keyword scoring with phrase-match boost

## Project Structure
```
ai-study-assistant/
├── backend/
│   └── app.py          # RAG pipeline + HTTP server
├── frontend/
│   └── index.html      # Single-page UI
├── data/               # Auto-created: chunks.json, index.json
└── README.md
```
