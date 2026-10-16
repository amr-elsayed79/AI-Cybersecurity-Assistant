# AI Cybersecurity Assistant — API Integration Guide & Contract

**Target Audience:** Team Leader & Frontend/Scanner Integration Developers  
**Service Version:** `0.2.0`  
**Current Architecture:** Standalone FastAPI Service with RAG Pipeline & Deterministic Attribution Guardrails  
**Local Demo Dashboard:** `http://127.0.0.1:8000/dashboard/`  

---

## 1. Project Purpose & System Boundaries

The **AI Cybersecurity Assistant** is an independent, standalone service that answers general software security, vulnerability mechanics, detection methodology, and defensive remediation questions using a curated Master Vulnerability Knowledge Base (KB).

### Key Architectural Boundaries:
- **No Scanner Responsibility:** This service **does not execute security scans**, does not probe live web applications, does not parse active scanner reports, and does not modify the teammate's vulnerability scanner codebase.
- **Knowledge Base Scope:** The Knowledge Base covers the **119 Active scanner scenarios**, normalized into **74 canonical vulnerability records** and **588 indexed semantic chunks**. The 74 Deferred scanner scenarios are explicitly out-of-scope and excluded.
- **Grounding vs Diagnosis:** Generic Knowledge Base information **cannot verify** whether a specific live URL or website is vulnerable. For site-specific questions, the assistant prepends a mandatory assessment disclaimer and provides general educational principles.

---

## 2. Local Prerequisites & Installation

The verified runtime environment is **Python 3.13.x 64-bit on Windows** (specifically tested on Python 3.13.9 64-bit). Compatibility with other Python versions (including Python 3.10–3.12) has not been verified with the current pinned dependency set.

### Setup in Windows PowerShell:

```powershell
# 1. Clone / Navigate to project repository
cd "D:\AI Cybersecurity Assistant"

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements-phase3.txt
pip install -r requirements-phase4.txt
```

---

## 3. Configuration & Safe Secret Handling

Configuration is managed via environment variables and loaded through Pydantic BaseSettings in `app/config.py`.

### Safe `.env` Management:
- Copy `.env.example` to create your local `.env`:
  ```powershell
  Copy-Item .env.example .env
  ```
- **Git Safety Guarantee:** `.env` is permanently excluded from Git tracking via `.gitignore` (`*.env`). **Never commit, push, print, or upload the `.env` file or any API keys.**

### Provider Modes:
1. **Mock Provider (Default / Offline Testing):**
   ```ini
   LLM_PROVIDER="mock"
   ```
   Requires no API key, zero network calls, runs 100% offline, and returns deterministic mock responses for pipeline flow validation.
2. **Groq Provider (Live Inference):**
   ```ini
   LLM_PROVIDER="groq"
   GROQ_API_KEY="gsk_..."
   GROQ_MODEL="openai/gpt-oss-120b"
   GROQ_TEMPERATURE=0.1
   GROQ_MAX_TOKENS=1024
   GROQ_TIMEOUT_SECONDS=20.0
   GROQ_MAX_RETRIES=2
   ```

---

## 4. Vector Database Indexing

The assistant uses an embedded Qdrant vector database (`data/qdrant_db`) and Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions).

If the vector store needs to be initialized or rebuilt from the Master KB:

```powershell
# Rebuild Qdrant vector collection from Master KB (588 chunks)
python scripts/index_kb.py --recreate
```

*Expected output:* `Total chunks generated: 588`, `Upserted: 588 points`. Master KB SHA-256 integrity is checked automatically before indexing.

---

## 5. Starting the Service

Launch the FastAPI application with Uvicorn:

```powershell
# Start Uvicorn server bound to localhost
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive OpenAPI Documentation (Swagger UI):** `http://127.0.0.1:8000/docs`
- **Alternative Documentation (ReDoc):** `http://127.0.0.1:8000/redoc`
- **Demo Dashboard:** `http://127.0.0.1:8000/dashboard/`

---

## 6. Health & Diagnostic Endpoints

### `GET /api/v1/health` (Alias: `GET /health`)
Inspects system health, Master KB hash integrity, Qdrant index count, and LLM provider configuration.

#### Response Schema (`HealthResponse`):
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "master_kb_checksum_verified": true,
  "vector_store_status": "ready",
  "indexed_point_count": 588,
  "llm_provider": "mock",
  "llm_configured": true
}
```

#### Field Explanations:
- `status`: `"healthy"` if vector store is ready (>= 588 points), KB hash matches baseline, and LLM provider is configured; `"degraded"` otherwise (HTTP 503).
- `master_kb_checksum_verified`: `true` if `data/master/ai_security_vulnerability_kb_master_v0.2.json` matches baseline SHA-256 (`d59af97e75dce8540f5dfaf1ba2d4e4afbcfe99c2ac697779bb5e9c6c0e99471`).
- `vector_store_status`: Detailed store connectivity/indexing state:
  - `"ready"`: Fully indexed with baseline points (>= 588 points).
  - `"partial"`: Reachable but partially indexed (1 to 587 points).
  - `"empty"`: Reachable collection with 0 indexed points.
  - `"unreachable"`: Exception encountered attempting to connect or query store.
  - `"unavailable"`: Store instance could not be initialized.
- `indexed_point_count`: Total points in Qdrant collection (expected: `588`).
- `llm_provider`: Configured provider name (`"mock"` or `"groq"`).
- `llm_configured`: `true` if required credentials exist for the selected provider.

---

## 7. Chat Endpoint Contract

### `POST /api/v1/chat` (Alias: `POST /api/chat`)

Submits a security query to the RAG pipeline with optional conversation continuity.

### Request Body (`ChatRequest`):
```json
{
  "message": "What is SQL injection, and what are two defensive ways to prevent it?",
  "conversation_id": "optional-uuid4-or-session-slug"
}
```

#### Request Validation Rules:
- `message` (or alias `query`): String, required, 1 to 2,000 characters. Cannot be empty or whitespace-only.
- `conversation_id`: String, optional, 1 to 64 characters, matching regex `^[a-zA-Z0-9_\-]{1,64}$`.
  - If omitted or `null`: The server generates a fresh UUID4 session identifier.
  - If provided: Continues the bounded in-memory conversation history for that session.
  - If invalid format (e.g. contains spaces, symbols, or exceeds 64 chars): Returns `HTTP 422 Unprocessable Content`.

### Response Body (`ChatAPIResponse`):
```json
{
  "answer": "SQL injection is an injection-type vulnerability... [E1]. Defenses include parameterized queries [E3]...",
  "grounded": true,
  "grounding_status": "grounded",
  "is_site_specific": false,
  "language": "en",
  "retrieved_sources": [
    {
      "chunk_id": "sql-injection_overview_and_mechanism",
      "vulnerability_id": "sql-injection",
      "name": "SQL Injection",
      "section": "overview_and_mechanism",
      "cwe": ["CWE-89"],
      "owasp": ["A05:2025 - Injection"],
      "score": 0.7801,
      "metadata": {}
    }
  ],
  "cited_sources": [
    {
      "chunk_id": "sql-injection_overview_and_mechanism",
      "vulnerability_id": "sql-injection",
      "name": "SQL Injection",
      "section": "overview_and_mechanism",
      "cwe": ["CWE-89"],
      "owasp": ["A05:2025 - Injection"],
      "score": 0.7801,
      "metadata": {}
    }
  ],
  "retrieval_latency_ms": 109.32,
  "llm_latency_ms": 1165.41,
  "total_latency_ms": 1330.45,
  "conversation_id": "8f3e2b1a-4c5d-6e7f-8a9b-0c1d2e3f4a5b"
}
```

---

## 8. HTTP Status Codes & Error Behaviors

| HTTP Status Code | Scenario | Response Payload Format |
| :--- | :--- | :--- |
| **`200 OK`** | Successful completion (grounded, refusal, disclaimer, or truncation notice). | Complete `ChatAPIResponse` JSON. |
| **`422 Unprocessable Content`** | Request validation error: empty query, whitespace-only query, query exceeding 2,000 characters, missing message body, malformed JSON, or invalid `conversation_id` regex violation. Intercepted and handled at FastAPI/Pydantic request schema layer. | `{"detail": "Validation error", "errors": ["body -> message: Query string cannot be empty or whitespace-only."]}` |
| **`400 Bad Request`** | Downstream pipeline query validation error fallback. Reached only if a `ValueError` is raised during internal pipeline processing rather than at the request schema layer. | `{"detail": "Query exceeds maximum allowed length of 2000 characters..."}` |
| **`503 Service Unavailable`** | LLM provider timeout, rate limit exhaustion, or network failure. | `{"detail": "LLM service is temporarily unavailable. Please try again later."}` (internal stack masked). |
| **`500 Internal Error`** | Unhandled internal server exception. | `{"detail": "An unexpected error occurred while processing your request."}` (masked for security). |

---

## 9. Grounding Status (`grounding_status`) Reference

The assistant enforces deterministic guardrails. The `grounding_status` field signals the exact operational outcome:

| Status Enum Value | `grounded` Value | Meaning & Pipeline Action |
| :--- | :---: | :--- |
| **`"grounded"`** | `true` | All citations match injected Master KB evidence. No unverified CWE/CVE identifiers detected. Full verified answer provided. |
| **`"site_qualified"`** | `true` | The query asked about a specific target URL or application (e.g., `https://secure.example.com`). The server prepends the mandatory assessment disclaimer and provides general educational principles. |
| **`"insufficient_evidence"`** | `false` | Retrieved KB chunks lacked relevance to the question, or model triggered standard refusal, or content filters triggered. Standard honest refusal returned. |
| **`"empty_retrieval"`** | `false` | Vector and lexical retrieval returned zero relevant chunks. Standard refusal returned with zero LLM inference calls. |
| **`"attribution_failed"`** | `false` | Model hallucinated an unsupported evidence tag (e.g. `[E99]`) or cited unverified CWE/CVE numbers. Answer suppressed for safety assurance. |
| **`"incomplete_truncated"`** | `false` | Generation reached the provider token ceiling (`finish_reason="length"`). Partial narrative was suppressed to prevent incomplete or misleading security guidance. |

---

## 10. Citation Semantics (Retrieved vs Cited)

- **`retrieved_sources`:** The candidate chunks injected into the model's prompt context (up to `max_context_chunks=4`).
- **`cited_sources`:** The exact subset of chunks whose labels (`[E1]`, `[ E1 ]`) were cited in the answer narrative.
- **Verification Guarantee:** Every item in `cited_sources` is verified against the server-side `EvidenceRegistry`.
- **Entailment Boundary:** Backend attribution proves that cited labels correspond to genuine injected chunks; it does not constitute formal mathematical proof of sentence-level natural language entailment.

---

## 11. Code Integration Examples

### Python (using `httpx` or `requests`):

```python
import httpx

API_URL = "http://127.0.0.1:8000/api/v1/chat"

payload = {
    "message": "What is OS Command Injection, and how do I prevent it?",
    "conversation_id": None  # Set to an existing UUID to maintain multi-turn context
}

response = httpx.post(API_URL, json=payload, timeout=30.0)
data = response.json()

print(f"Status: {data['grounding_status']}")
print(f"Session: {data['conversation_id']}")
print(f"Answer:\n{data['answer']}")
print(f"\nCited Chunks ({len(data['cited_sources'])}):")
for src in data['cited_sources']:
    print(f" - [{src['chunk_id']}] {src['name']} (Section: {src['section']})")
```

### JavaScript (`fetch` / ES6):

```javascript
async function askSecurityAssistant(userPrompt, conversationId = null) {
    const response = await fetch('http://127.0.0.1:8000/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: userPrompt,
            conversation_id: conversationId
        })
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`API Error ${response.status}: ${errorData.detail || 'Request failed'}`);
    }

    const result = await response.json();
    return {
        answer: result.answer,
        status: result.grounding_status,
        conversationId: result.conversation_id,
        citations: result.cited_sources
    };
}
```

### cURL:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "How does XSS prevention work?", "conversation_id": "test-session-1"}'
```

---

## 12. Instructions for Integrating into the Scanner Dashboard

To integrate this assistant into the teammate's vulnerability-scanner website without modifying scanner logic:

1. **Decoupled Architecture:** Keep the scanner backend and this assistant running as separate services. The scanner website frontend can embed the assistant via an iframe or invoke the `/api/v1/chat` endpoint directly.
2. **CORS Configuration:**
   - By default, `CORS_ALLOW_ORIGINS` in `app/config.py` already includes:
     - `http://localhost:3000`
     - `http://127.0.0.1:3000`
     - `http://localhost:8000`
     - `http://127.0.0.1:8000`
   - If the scanner website runs on an alternative origin or port (e.g. `http://localhost:5173`), update `CORS_ALLOW_ORIGINS` in `.env` or `app/config.py`:
     ```ini
     CORS_ALLOW_ORIGINS=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000", "http://localhost:5173"]
     ```
3. **Recommended Production Reverse Proxy:**
   - Instead of exposing two separate web origins to the browser, configure an Nginx or Caddy reverse proxy on the host:
     - `/` and `/scanner/*` → Routes to teammate's scanner dashboard.
     - `/assistant/api/*` → Proxied to `http://127.0.0.1:8000/api/*`.
     - `/assistant/dashboard/*` → Proxied to `http://127.0.0.1:8000/dashboard/*`.
   - This achieves same-origin API calls without requiring cross-origin credentials.

---

## 13. Conversation Session & Memory Semantics

- **In-Memory & Ephemeral:** Conversation history is stored in an in-memory `OrderedDict` (`InMemoryConversationStore`). All sessions are lost immediately upon process restart.
- **Bounded Capacity:** Each session retains at most **6 turns** (3 user prompts + 3 assistant responses) in FIFO order. The store holds at most **1,000 active conversations** before LRU eviction.
- **Single Process Only:** State is protected with a process-level `threading.Lock`. It is not shared across multi-process workers (`uvicorn --workers > 1`).
- **No Authenticated Ownership:** Session isolation relies on random 128-bit UUID4 unguessability. Anyone who knows a `conversation_id` can append messages to it. For multi-tenant production, an authentication layer must bind conversation IDs to authenticated user identities.

---

## 14. Deployment & Security Boundaries

Before exposing the API beyond `127.0.0.1`, the following operational boundaries must be respected:

1. **Localhost Binding:** Must remain bound to `127.0.0.1`. Do not bind to `0.0.0.0` or expose port 8000 to public networks without a secure reverse proxy.
2. **Authentication:** The API does not currently validate API keys or JWT tokens on incoming chat requests. Add an auth gateway prior to public deployment.
3. **Rate Limiting:** No rate limiting is enforced in code. Rapid requests will exhaust the Groq API rate limit (HTTP 429). An API gateway or rate limiter (e.g., `slowapi`) is required for public endpoints.
4. **Qdrant Embedded File Lock:** Qdrant runs in embedded mode (`data/qdrant_db`). It acquires an OS file lock (`.lock`). **Running Uvicorn with multiple worker processes will cause lock acquisition crashes.** To scale horizontally, run Qdrant as a standalone server over HTTP.

---

## 15. Known Limitations

1. **Active Scenarios Representation:** The 119 Active scanner scenarios are represented through 74 canonical vulnerability records. The Master KB does not contain a 1-to-1 scenario-to-chunk mapping table.
2. **Deferred Scenarios:** The 74 Deferred scanner scenarios are explicitly omitted.
3. **Synthetic / Mock Mode:** In `LLM_PROVIDER=mock` mode, responses validate software flow, schema serialization, and guardrails only; they do not assess factual reasoning.
