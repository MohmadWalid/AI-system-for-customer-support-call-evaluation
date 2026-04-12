# Phase 4: RAG Implementation TODO

## Phase 1: Setup (Data Ingestion & Indexing)
- [x] Define mock/fake policy dataset (tags: advice, disclosure, resolution, closing, objection_handling)
- [x] Implement text chunking logic for policies (if necessary)
- [x] Load local embedding model (`sentence-transformers: all-MiniLM-L6-v2`)
- [x] Generate embeddings for policy chunks
- [x] Initialize FAISS index
- [x] Store policy embeddings into FAISS
- [x] Create mapping between FAISS indices and original policy text/metadata

*Implemented in: `src/setup_index.py`, `data/policies.py`*

## Phase 2: Runtime (Retrieval & Evaluation)
- [x] Define mock/fake transcript data format
- [x] Load the pre-built FAISS index and policy mappings
- [x] Implement utterance processing pipeline:
  - [x] Embed the utterance using the local `all-MiniLM-L6-v2` model
  - [x] Query FAISS to retrieve top-k (e.g., k=3) policy indices
  - [x] Convert retrieved indices back to original policy text
  - [x] Evaluate similarity score against a defined threshold
  - [x] Assign confidence level: "High" (score ≥ threshold) or "Low" (score < threshold)
- [x] Integrate closely with Groq API ("Llama 4 Scout"):
  - [x] Construct LLM prompt containing the utterance and the retrieved policies
  - [x] Send request to Groq API
- [x] Parse and format LLM output to match required schema:
  - [x] Violation (Yes/No)
  - [x] Violated Policy
  - [x] Matched Policy ID / Name
  - [x] Reason
  - [x] Evidence
  - [x] Confidence (High/Low)

*Implemented in: `src/runtime_rag.py`, `data/transcripts.py`*

## Phase 3: Configuration & Dependencies
- [x] Setup `requirements.txt` (dependencies: `sentence-transformers`, `faiss-cpu`, `groq`, etc.)
- [x] Setup `.env` file (for `GROQ_API_KEY`, `SIMILARITY_THRESHOLD`, `TOP_K`)

*Implemented in: `requirements.txt`, `.env.example`, `config.py`*
