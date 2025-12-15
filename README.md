# Llama Knowledge Base AI Agent (Multilingual)

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end **AI Agent** built with **Flask**, **Groq-hosted LLaMA models**, and **hybrid retrieval (semantic + keyword)**, designed to answer questions **strictly from a user-provided Microsoft Word (.docx) knowledge base**.

This version adds **multilingual support (English + Kiswahili)**, including:
- Text and audio prompts in **English or Kiswahili**
- **Kiswahili → English translation for retrieval** (recommended for English KBs)
- Responses returned in the **selected language**
- Optional **browser-based location permission** (“Use my location”) for location-aware questions

> **Important grounding rule:** The agent answers only using the **retrieved KB excerpts** (top-ranked chunks). If the answer is not supported by the KB excerpts, it will say so.

---

## Project Overview

**Key features**
- Upload a Word document as a Knowledge Base (KB)
- **Hybrid search (semantic + keyword)** for stronger table/term matching
- Grounded answers using Groq-hosted LLaMA models
- **Voice input** (browser recording, `.wav`, `.m4a`)
- **Voice transcription** using **Groq’s hosted Whisper API**
- Persistent, versioned KB storage (swap KBs)
- Modern web chat UI (templates + static)
- Embeddable bottom-right chat widget
- OpenAPI / Swagger documentation
- MIT licensed (dual attribution)

---

## Retrieval Architecture (Hybrid – Important)

This agent uses **hybrid retrieval**, combining semantic similarity with keyword matching.

### Scoring logic

```
final_score = α * semantic_score + (1 - α) * keyword_score
```

- Semantic search captures meaning and intent
- Keyword search captures exact terms, tables, IDs, and regions
- Hybrid retrieval significantly improves accuracy for structured documents
- The hybrid approach reduces hallucinations by enforcing KB grounding

---

## Multilingual Behavior (English + Kiswahili)

### Why translation is needed
Most KBs (like the Digital Hubs status tables) are authored in **English**. If a user asks in **Kiswahili**, retrieval can miss relevant chunks.  
To fix that, the agent performs:

- **Kiswahili query → English translation (retrieval-only)**
- Run **hybrid retrieval** on the English version
- Generate the final answer in **Kiswahili** (or English) depending on the selected language

### What is translated?
- ✅ The *query used for retrieval* (when `language=sw`)
- ❌ The KB is not translated or modified
- ✅ The final answer language is enforced by prompting

---

## System Architecture Diagram (For Proposals & Integrators)

```mermaid
flowchart LR
    U[User / Portal / Mobile App] -->|Text or Audio| A[Flask API + Web UI]

    %% Language selection / routing
    A --> LS[Language Selector<br/>(English / Kiswahili)]

    %% Audio transcription
    A -->|Audio Input| W[Groq Whisper API<br/>Speech-to-Text]
    W -->|Transcribed Text| Q[User Query Text]

    %% Text path
    A -->|Text Query| Q[User Query Text]

    %% Translation (retrieval-only)
    Q -->|If Kiswahili| T[Groq LLaMA (small)<br/>Translate SW → EN<br/>(retrieval-only)]
    Q -->|If English| RQ[Retrieval Query (EN)]
    T --> RQ[Retrieval Query (EN)]

    %% Hybrid retrieval
    RQ --> R[Hybrid Retrieval Engine]
    R --> S[Semantic Search<br/>Vector Embeddings]
    R --> K[Keyword Search<br/>TF-IDF Index]
    S --> H[Hybrid Scoring & Ranking]
    K --> H[Hybrid Scoring & Ranking]

    %% LLM answer generation
    H -->|Top-K KB Chunks| L[LLaMA Model via Groq API<br/>Grounded Answer]
    LS -->|Selected language| L

    L -->|Answer (EN or SW)| A
    A -->|UI / JSON| UI[Web UI / Embedded Chat Widget]
```

### Architectural notes
- The LLaMA model only sees **top-ranked KB excerpts**, never the full document
- Audio transcription uses **Groq-hosted Whisper**
- Kiswahili queries translate to English for retrieval when KB is English
- Hallucination is mitigated by **strict KB-grounded prompting**
- Supports web UI, iframe embedding, and direct API access
- Suitable for **enterprise and government deployments**

---

## Directory Structure (Project Root)

Your project layout remains:

- `app.py`
- `templates/`
- `static/`
- `kb_store/`
- `.env`, `.gitignore`, `requirements.txt`, etc.

---

## Clone the Repository

```bash
git clone https://github.com/gaswani/llama-kb-agent-multilingual.git
cd llama-kb-agent-multilingual
```

---

## Python & Virtual Environment Setup

### Required Python Version
```
Python 3.11.9
```

### macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Environment Variables (.env)

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Hybrid retrieval tuning
HYBRID_SEARCH=true
HYBRID_ALPHA=0.70
HYBRID_THRESHOLD=0.35

# Optional: Nominatim reverse-geocode (for "Use my location")
NOMINATIM_URL=https://nominatim.openstreetmap.org/reverse
NOMINATIM_USER_AGENT=llama-kb-agent/1.0
```

> If you added a dedicated translation model setting in your app, document it here too (e.g. `GROQ_TRANSLATION_MODEL=...`).

---

## Run the Application

```bash
python app.py
```

Open in your browser:

```
http://127.0.0.1:8000/
```

---

## API Documentation (OpenAPI / Swagger)

- **Swagger UI (interactive documentation):**
```
http://127.0.0.1:8000/docs
```

- **OpenAPI specification (JSON):**
```
http://127.0.0.1:8000/openapi.json
```

---

## OpenAPI Screenshots (for Proposals & Documentation)

Recommended screenshots:
- Swagger UI – endpoint overview
- Chat endpoints (`/chat_text`, `/chat`)
- Knowledge base management (`/upload_kb`, `/kb/*`)

Suggested structure:
```
docs/
└── screenshots/
    ├── swagger_overview.png
    ├── chat_endpoints.png
    └── kb_management.png
```

---

## Upload Knowledge Base (KB)

### macOS / Linux

```bash
curl -X POST http://localhost:8000/upload_kb \
  -F "file=@your_document.docx"
```

### Windows (PowerShell or Command Prompt)

```powershell
curl.exe -X POST http://localhost:8000/upload_kb `
  -F "file=@your_document.docx"
```

### Windows (PowerShell-native)

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/upload_kb" `
  -Method Post `
  -Form @{ file = Get-Item "your_document.docx" }
```

> **Note:** In PowerShell, `curl` is an alias for `Invoke-WebRequest` and does **not** support `-F`.  
> Always use `curl.exe` or the PowerShell-native example above.

---

## KB Reloading (Hybrid Search)

Hybrid retrieval builds an in-memory keyword index.

```bash
curl -X POST http://127.0.0.1:8000/kb/reload
```

---

## Test Queries

### English

```bash
curl -X POST http://127.0.0.1:8000/chat_text \
  -H "Content-Type: application/json" \
  -d '{"message":"Show me one row of hub details including region, hub name, and status.", "language":"en"}'
```

### Kiswahili (retrieval translated SW → EN internally)

```bash
curl -X POST http://127.0.0.1:8000/chat_text \
  -H "Content-Type: application/json" \
  -d '{"message":"Nionyeshe mstari mmoja wa maelezo ya hub: eneo, jina la hub na hali yake.", "language":"sw"}'
```

---

## Audio Testing (API)

> Your app supports audio via the `/chat` endpoint as multipart form-data.

### macOS / Linux

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -F "audio=@test_audio_prompt.m4a" \
  -F "language=sw"
```

### Windows (PowerShell)

```powershell
curl.exe -X POST http://127.0.0.1:8000/chat `
  -F "audio=@test_audio_prompt.m4a" `
  -F "language=sw"
```

---

## Browser UI: Language + Location

- Use the language dropdown to choose **English** or **Kiswahili**
- Click **Use my location** (optional):
  - Your browser will ask for permission
  - The UI sends `{lat, lon}` to `/geo/reverse`
  - The agent uses the resulting county/region as an additional retrieval hint

---

## Embedding the Agent in a Web Portal

### Option A (Recommended): Floating Bottom-Right Widget

A ready-made example is included as:

```
iframe_widget.html
```

Replace:

```html
src="https://YOUR_AGENT_DOMAIN_OR_IP/"
```

With:

```
http://127.0.0.1:8000/        (local testing)
https://agent.yourcompany.com (production)
```

---

## For Integrators (Portal & App Teams)

### Integration options
- Iframe embed (fastest)
- Direct REST API calls (most flexible)
- Audio-first workflows (`/chat`)

### Enterprise / Government deployment notes
- Protect administrative endpoints (KB upload/swap) behind auth
- Restrict `/docs` in production
- Deploy behind HTTPS (reverse proxy)
- Enable logging + monitoring
- Consider rate limiting for public portals

---

## License

MIT License  

© 2025 Gideon Aswani / Pathways Technologies Ltd

---

## Third-Party Model Notice

This project uses **LLaMA-family models served via the Groq API**.
No model weights are distributed.

Usage is subject to:
- Groq API Terms of Service
- Meta LLaMA Acceptable Use Policy
