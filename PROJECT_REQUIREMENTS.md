# PROJECT REQUIREMENTS — AI Cybersecurity Assistant

## 1. Project Goal

Build a production-structured RAG-based AI Cybersecurity Assistant.

The assistant answers general cybersecurity and web-application-security questions using a curated vulnerability Knowledge Base.

IMPORTANT:
- The Knowledge Base is the source of truth for vulnerability knowledge.
- Do NOT train/fine-tune an LLM for this project.
- Use RAG: retrieve relevant knowledge, then give it to an LLM as context.
- Do NOT use scanner reports as the chatbot's Knowledge Base.
- Do NOT include Deferred scanner scenarios.
- The current scope is the 119 ACTIVE scanner scenarios, normalized into canonical vulnerability records.
- The current Master KB contains 74 canonical vulnerability records.

## 2. Source Data

Primary source:
`ai_security_vulnerability_kb_master_v0.2.json`

Optional processed reference:
`ai_security_rag_chunks_v0.1.jsonl`

RULE:
The Master JSON is the source of truth.
The RAG chunk JSONL should be reproducible by the ingestion pipeline from the Master JSON.
Do not maintain two independent knowledge sources.

## 3. Main User Capabilities

The chatbot should answer questions such as:

### General vulnerability questions
- What is SQL Injection?
- How does XSS work?
- What causes SSRF?
- What is IDOR?

### Risk and severity
- How dangerous is this vulnerability?
- What is the impact?
- Why can this be Critical?
- What factors affect severity?

### Detection
- How can I detect this vulnerability?
- What should I look for in code?
- What are common indicators?
- How can a security tester assess it?

### Prevention
- How can I prevent it?
- What secure coding practices should I use?

### Remediation
- How do I fix it?
- How can I verify that the fix worked?

### Security metadata
- What is the CWE?
- What is the OWASP category?
- Which technologies are commonly affected?

### Comparisons
- SQL Injection vs NoSQL Injection
- XSS vs SQL Injection
- IDOR vs Broken Access Control
- JWT weaknesses vs session vulnerabilities

### General website/API security
The assistant may answer broader security questions when the retrieved KB contains enough information.

## 4. Important Scope Boundary

The chatbot must distinguish between:

A. General security knowledge
Example:
"How do I prevent SQL Injection?"

This can be answered from the Knowledge Base.

B. Website-specific security claims
Example:
"Is my website vulnerable to SQL Injection?"

The chatbot MUST NOT claim that a website is vulnerable based only on the generic Knowledge Base.

For site-specific claims, the system should say that runtime scan/code/site evidence is required.

## 5. RAG Architecture

Implement:

User
→ Chat API
→ Query preprocessing
→ Query embedding
→ Vector retrieval
→ Metadata filtering when useful
→ Reranking
→ Context assembly
→ LLM
→ Answer + citations

Use a clean modular architecture so components can be replaced later.

## 6. Recommended Technology Stack

Backend:
- Python
- FastAPI

RAG:
- Qdrant as the preferred vector database
- Sentence Transformers or another strong embedding model
- Optional reranker using a cross-encoder/reranking model

LLM:
- Keep the LLM provider configurable through environment variables.
- Do not hard-code API keys.
- Make the LLM adapter replaceable.

Frontend:
- Next.js / React
- Clean modern chat interface
- Arabic + English support
- RTL/LTR support

Configuration:
- `.env`
- `.env.example`

## 7. Project Structure

Prefer:

app/
├── main.py
├── config.py
├── api/
│   └── chat.py
├── rag/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── context_builder.py
│   └── pipeline.py
├── llm/
│   └── provider.py
├── prompts/
│   └── security_assistant.txt
├── models/
│   └── schemas.py
├── evaluation/
│   └── eval_questions.json
└── utils/

data/
├── master/
│   └── ai_security_vulnerability_kb_master_v0.2.json
└── processed/

frontend/
...

scripts/
├── ingest.py
├── rebuild_index.py
└── evaluate.py

tests/
├── test_chunking.py
├── test_retrieval.py
└── test_api.py

README.md
.env.example
requirements.txt
docker-compose.yml

You may improve this structure if there is a strong technical reason, but keep responsibilities separated.

## 8. Ingestion Pipeline

Build a reproducible ingestion command.

Example:

`python scripts/ingest.py`

It should:

1. Load Master JSON.
2. Validate schema.
3. Create semantic chunks.
4. Attach metadata.
5. Generate embeddings.
6. Create/update Qdrant collection.
7. Upsert vectors.
8. Report number of vulnerabilities/chunks/vectors.

Do not manually copy the RAG chunks into the code.

## 9. Chunking Rules

Do NOT create one huge chunk per vulnerability.

Preferred semantic sections:

- overview
- how_it_works
- root_cause
- variants
- severity
- impact
- risk_factors
- affected_technologies
- affected_languages
- affected_frameworks
- common_locations
- detection
- prevention
- remediation
- secure_coding
- common_mistakes
- cwe_owasp
- related_vulnerabilities
- comparisons
- examples

Keep related information together.

Every chunk must have metadata such as:

{
  "vulnerability_id": "...",
  "name": "...",
  "category": "...",
  "section": "...",
  "severity": "...",
  "cwe": [...],
  "owasp": [...]
}

## 10. Retrieval

Implement semantic vector retrieval.

Initial strategy:
- retrieve top-k candidates
- optionally filter by category/vulnerability/section
- rerank candidates
- send only the best context to the LLM

Make `top_k` configurable.

Do not retrieve the entire Knowledge Base.

## 11. Reranking

Implement the reranker as a replaceable module.

If the selected reranker is unavailable locally, provide a clean fallback to vector similarity retrieval.

The application must still run without a reranker.

## 12. LLM Prompt Rules

The system prompt should enforce:

1. You are an AI Cybersecurity Assistant.
2. Use retrieved Knowledge Base context as the primary factual source.
3. Do not invent facts, CWE IDs, OWASP mappings, or references.
4. If the context is insufficient, explicitly say that the available Knowledge Base does not contain enough information.
5. Never claim that a specific website is vulnerable without site-specific evidence.
6. Distinguish general educational information from site-specific findings.
7. Explain technical concepts clearly.
8. Answer in the user's language when practical.
9. Preserve technical names such as CWE, OWASP, XSS, SQLi, SSRF, etc.
10. Cite the retrieved knowledge chunks used for the answer.

## 13. Citations

Every factual answer based on retrieved KB context should expose source information.

Example:

Sources:
- SQL Injection — Prevention
- SQL Injection — CWE/OWASP

The API should return source metadata separately from the generated answer so the frontend can render citations.

## 14. Chat API

Implement at least:

POST `/api/chat`

Request:

{
  "message": "How can I prevent SQL Injection?",
  "conversation_id": "optional"
}

Response should include:

{
  "answer": "...",
  "sources": [
    {
      "chunk_id": "...",
      "vulnerability_id": "...",
      "name": "...",
      "section": "prevention"
    }
  ]
}

Add health endpoint:

GET `/health`

## 15. Conversation History

Support conversation IDs.

Do not send unlimited conversation history to the LLM.

Use a configurable history window.

The RAG retrieval should primarily depend on the current user question plus useful conversational context.

## 16. Safety and Accuracy

This is a defensive cybersecurity assistant.

Allowed:
- vulnerability explanations
- detection concepts
- secure coding
- remediation
- defensive testing methodology
- risk analysis

Avoid turning answers into weaponized exploit/payload libraries.

For dangerous requests, keep responses defensive and educational.

## 17. Evaluation Dataset

Create a small evaluation set before calling the chatbot complete.

Include questions covering:

- basic definitions
- causes
- severity
- impact
- detection
- prevention
- remediation
- CWE/OWASP
- comparisons
- ambiguous questions
- questions whose answer is NOT in the KB
- site-specific questions that must be rejected/qualified

At least 50 evaluation questions.

Measure:
- retrieval relevance
- source correctness
- answer groundedness
- refusal/qualification correctness
- citation correctness

## 18. No Hallucination Requirement

If retrieval returns weak or irrelevant context:

Do NOT force an answer from unrelated chunks.

Return a useful limitation such as:

"I don't have enough relevant information in the current security knowledge base to answer that reliably."

## 19. Developer Experience

Provide:

- README
- setup instructions
- `.env.example`
- requirements file
- startup commands
- ingestion command
- index rebuild command
- evaluation command
- API documentation
- clear error messages

The project should be easy for a student/developer to run locally.

## 20. Environment Variables

Use environment variables for:

- LLM API key
- LLM model
- embedding model
- Qdrant URL
- Qdrant API key if needed
- collection name
- retrieval top-k
- reranker configuration

Never commit secrets.

## 21. Docker

Provide Docker Compose if practical for:

- Qdrant
- backend
- optionally frontend

Local development without Docker should still be possible.

## 22. Testing

Write tests for:

- JSON loading
- schema validation
- chunk generation
- metadata generation
- embedding pipeline
- retrieval
- API response format
- citation generation
- insufficient-context behavior

## 23. Implementation Order

Do NOT attempt to generate everything blindly in one step.

Work in this order:

Phase 1:
Create project structure and configuration.

Phase 2:
Implement Knowledge Base validation and ingestion.

Phase 3:
Implement chunking and metadata.

Phase 4:
Implement embeddings and Qdrant.

Phase 5:
Implement retrieval.

Phase 6:
Implement reranking with fallback.

Phase 7:
Implement LLM provider adapter and security system prompt.

Phase 8:
Implement `/api/chat`.

Phase 9:
Implement citations and conversation history.

Phase 10:
Implement frontend.

Phase 11:
Implement evaluation.

Phase 12:
Run tests and fix errors.

After each phase:
- run tests
- verify files
- explain what was implemented
- do not silently skip failures

## 24. Important Development Rule

Before writing major code:

1. Inspect the existing project.
2. Inspect the Master JSON.
3. Understand its actual schema.
4. Do not invent fields that are not present.
5. Do not overwrite existing useful code without checking it.
6. Reuse good existing components when possible.

## 25. Definition of Done

The project is complete only when:

- Master JSON loads successfully.
- Ingestion is reproducible.
- Qdrant contains the indexed knowledge.
- Retrieval returns relevant chunks.
- Reranker works or cleanly falls back.
- LLM answers are grounded in retrieved context.
- Sources are returned.
- Site-specific claims are properly qualified.
- Unknown questions are handled safely.
- Frontend chat works.
- Arabic and English questions work.
- Tests pass.
- README explains the entire system.

## 26. First Task for the Coding Agent

Do NOT immediately write the whole application.

First:
1. Inspect the project directory.
2. Inspect `ai_security_vulnerability_kb_master_v0.2.json`.
3. Produce a short implementation plan.
4. Identify any schema/data-quality issues.
5. Then implement Phase 1 only.

Wait for confirmation before moving to the next major phase unless the user explicitly asks for full autonomous implementation.
