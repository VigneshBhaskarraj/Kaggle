# Nestlé HR Assistant — an evidence-driven RAG chatbot for HR policy documents

**Course-End Project:** *Crafting an AI-Powered HR Assistant: A Use Case for Nestlé's HR Policy Documents*

A conversational assistant that answers employees' questions about **The Nestlé Human Resources Policy** with
Retrieval-Augmented Generation — LangChain, OpenAI embeddings in Chroma DB, GPT-3.5 Turbo and a Gradio chat UI, as the
brief requires. What sets it apart is the approach: the design starts from the **failure modes of a naive RAG chatbot**,
answers each one, and then **measures** whether the answer works, so the chatbot ships with the configuration that won
the evaluation rather than the one that was assumed.

| Failure mode of naive RAG | Response | Measured by |
|---|---|---|
| Wrong passage retrieved | Section-aware chunking (the PDF's headings are extracted out of order — the parser repairs that) + hybrid BM25/vector retrieval with rank fusion | Hit@k / MRR on a gold question set, ablated over chunking × retrieval |
| Follow-ups lose context | Query rewriting into a standalone question before retrieval | Hit@4 on follow-ups, as typed vs. rewritten |
| Invented entitlements | Grounded prompt with an explicit *not-found* fallback + per-sentence groundedness check | Decline rate on out-of-scope questions, grounded-sentence ratio |
| No provenance | Section/page citations with quoted evidence + a confidence badge calibrated on the gold set | Shown on every answer |
| Quality unknown | 27 in-scope, 5 out-of-scope and 5 follow-up gold questions written from the policy; an evaluation harness that re-runs on every change | — |

```
 nestle_hr_policy.pdf ─PyPDFLoader─► pages ─structure recovery─► 10 sections ─split─► chunks (naive │ section-aware)
                                                                                           │
                                                   OpenAI embeddings ─► Chroma (cosine)   +   BM25 index
                                                                                           │
 question ─(rewrite if follow-up)─► hybrid retrieval (RRF) ─► prompt template ─► GPT-3.5 Turbo ─► answer
                                                                                           │
                       trust layer: confidence badge · groundedness · quoted evidence ─► Gradio ChatInterface
```

## Repository layout

| Path | Purpose |
|------|---------|
| `nestle_hr_assistant.ipynb` | **The deliverable.** The whole workflow with narrative: setup → PDF → structure → chunking → embeddings/Chroma → retrieval → prompt + GPT-3.5 Turbo chain → trust layer → gold set → evaluation & ablations → Gradio UI → conclusion. |
| `hr_assistant.py` | The same code as a module (the notebook embeds it cell by cell), used by the app and the tests. |
| `app.py` | Serves the chatbot outside the notebook: `python app.py [--strategy …] [--mode …] [--k …] [--no-rewrite] [--share]`. |
| `data/nestle_hr_policy.pdf` | *The Nestlé Human Resources Policy* (2012), Nestlé's public document ([source](https://www.nestle.com/sites/default/files/asset-library/documents/jobs/the_nestle_hr_policy_pdf_2012.pdf)); its SHA-256 is verified in the notebook. |
| `tests/test_offline.py` | Offline test suite — no API key needed (see below). |
| `requirements.txt` | Pinned, tested versions (Python 3.11). |
| `.env.example` | Template for `OPENAI_API_KEY` and optional model overrides. |

## Quick start

```bash
git clone https://github.com/VigneshBhaskarraj/Kaggle.git && cd Kaggle
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # paste your OpenAI API key into .env
```

**Run the notebook** (the submission):

```bash
pip install jupyter
jupyter notebook nestle_hr_assistant.ipynb             # Run All — the chat UI renders in the last cell
```

**Or run the chatbot as an app:**

```bash
python app.py                # http://127.0.0.1:7860 — section-aware chunking + hybrid retrieval
python app.py --share        # also creates a temporary public Gradio link
```

**Google Colab:** upload the repository (or `!git clone` it) — or just the notebook on its own: it is self-contained.
The first cell installs the pinned dependencies, the key cell prompts for the API key securely, and if
`data/nestle_hr_policy.pdf` is absent the notebook downloads the published PDF and verifies its checksum. The full run makes a few dozen GPT-3.5 Turbo
calls over a 14 000-character document — a few cents.

## How it works

1. **Load** — `PyPDFLoader` turns the 8-page PDF into page documents; the file's checksum is verified.
2. **Recover the structure** — inspecting the extraction showed a running header, hyphenated line breaks and section
   headings emitted *after* their body text. A small parser repairs all three and recovers the policy's 10 sections and
   their *"Corporate policy: …"* cross-references, validated against the table of contents.
3. **Chunk, two ways** — the textbook split over raw pages (20 chunks) and a section-aware split of identical size
   (19 chunks) whose chunks are prefixed with their section title. Same size, so the comparison isolates structure.
4. **Index** — `text-embedding-3-small` vectors in persisted Chroma collections (cosine space, one collection per
   strategy and embedding model) plus a BM25 index.
5. **Retrieve** — BM25, vector or **hybrid** (Reciprocal Rank Fusion); follow-up questions are first rewritten into
   standalone questions using the conversation history.
6. **Answer** — a prompt template fixing persona, grounding rules and an explicit fallback, sent with the top-k chunks
   to `gpt-3.5-turbo` (`temperature=0`; falls back to `gpt-4o-mini` only if the model has been retired for the key).
7. **Trust layer** — confidence badge from the best cosine similarity (thresholds calibrated on the gold set),
   per-sentence groundedness, and quoted evidence with section and page under every answer.
8. **Evaluate** — the chunking × retrieval grid, a k-sweep, the rewriting ablation and end-to-end answer metrics; the
   findings are generated from the numbers and the chatbot uses the winning configuration.

Models can be overridden with `LLM_MODEL=…` / `EMBEDDING_MODEL=…` (see `.env.example`); the index is rebuilt
automatically because the collection name includes the embedding model.

## Verifying without an API key

```bash
python tests/test_offline.py        # or: python -m pytest tests/
```

PDF loading, section parsing, chunking, BM25 retrieval and Chroma indexing run for real; OpenAI embeddings and the chat
model are replaced by LangChain's deterministic fakes. Besides checking the wiring and the evaluation harness, the suite
asserts a genuine retrieval-quality bar: **BM25 alone must reach Hit@4 ≥ 0.85 on the gold set** (currently 0.96 with
section-aware chunking vs. 0.93 naive — the first measured win for respecting the document's structure).

## Submitting to the LMS

1. Run the notebook top-to-bottom **with your API key** so every cell shows its output — the ablation tables, the
   generated findings and the chat UI are the point of the submission.
2. Read the *Findings* cell: it describes that run's numbers. Adjust the conclusion if the results surprise you.
3. Save and upload `nestle_hr_assistant.ipynb`.

## Notes

* **Document provenance.** The course materials describe the task but do not ship the policy PDF, so the official
  document published by Nestlé is used, with its checksum recorded for reproducibility.
* `.env` and the `chroma_db/` index are git-ignored — never commit API keys.
