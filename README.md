# Nestlé HR Assistant — an AI-powered chatbot for HR policy documents

**Course-End Project:** *Crafting an AI-Powered HR Assistant: A Use Case for Nestlé's HR Policy Documents*

A conversational chatbot that answers employees' questions about **The Nestlé Human Resources Policy** by retrieving the
relevant passages from the PDF and letting **GPT-3.5 Turbo** compose an answer grounded in them — Retrieval-Augmented
Generation (RAG) with LangChain, OpenAI embeddings, Chroma DB and a Gradio chat UI.

```
 nestle_hr_policy.pdf ──PyPDFLoader──► pages ──RecursiveCharacterTextSplitter──► chunks
                                                                                   │
                                                             OpenAI embeddings ──► Chroma vector store (persisted)
                                                                                   │
 user question ──embed──► top-k similar chunks ──► prompt template ──► GPT-3.5 Turbo ──► grounded answer + page citations
                                                                                                        │
                                                                                            Gradio ChatInterface
```

## Repository layout

| Path | Purpose |
|------|---------|
| `nestle_hr_assistant.ipynb` | **The deliverable.** Step-by-step notebook covering the whole workflow (setup → PDF → chunks → embeddings/Chroma → GPT-3.5 Turbo QA → prompt template → Gradio UI). |
| `app.py` | The same pipeline as a standalone script — run or deploy the chatbot without a notebook. |
| `data/nestle_hr_policy.pdf` | *The Nestlé Human Resources Policy* (Nestlé's public document, included for educational use — [source](https://www.nestle.com/sites/default/files/asset-library/documents/jobs/the_nestle_hr_policy_pdf_2012.pdf)). |
| `tests/test_offline.py` | End-to-end test of the pipeline with fake embeddings/LLM — verifies the wiring without an API key. |
| `requirements.txt` | Pinned, tested dependency versions (Python 3.11). |
| `.env.example` | Template for the `OPENAI_API_KEY` configuration. |

## Quick start

```bash
git clone https://github.com/VigneshBhaskarraj/Kaggle.git && cd Kaggle
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                                   # then paste your OpenAI API key into .env
```

**Run the notebook** (the submission):

```bash
pip install jupyter
jupyter notebook nestle_hr_assistant.ipynb             # Run All — the Gradio chat UI renders in the last cell
```

**Or run the chatbot as an app:**

```bash
python app.py                # http://127.0.0.1:7860
python app.py --share        # also creates a temporary public Gradio link
```

**Google Colab:** upload the repository (or `!git clone` it), open the notebook and *Run all*. The first cell installs the
pinned dependencies and the key cell prompts for your API key securely.

## How it works

1. **Load** — `PyPDFLoader` turns the PDF into one document per page (8 pages, page numbers kept as metadata).
2. **Split** — `RecursiveCharacterTextSplitter` (1 000 characters, 150 overlap) produces 20 coherent chunks.
3. **Embed & index** — `text-embedding-3-small` vectors are stored in a persisted Chroma collection (`chroma_db/`),
   which is re-used on later runs so embeddings are paid for once.
4. **Retrieve & answer** — the top-4 chunks for a question are placed into a prompt template and sent to
   `gpt-3.5-turbo` (`temperature=0`). The prompt fixes the HR-assistant persona, forbids inventing policies, defines an
   explicit fallback when the document does not cover a topic, and includes the conversation history so follow-up
   questions work.
5. **Chat UI** — `gr.ChatInterface` with example questions; every answer cites the policy pages it came from.

Models can be overridden through environment variables, e.g. `LLM_MODEL=gpt-4o-mini` or
`EMBEDDING_MODEL=text-embedding-3-large` (see `.env.example`).

## Verifying without an API key

```bash
python tests/test_offline.py
```

Runs the real PDF loading, chunking and Chroma indexing with LangChain's deterministic fake embeddings and a fake chat
model, then checks the chain output, the Gradio callback (citations and the not-found path) and the UI construction.

## Submitting to the LMS

1. Run the notebook top-to-bottom **with your API key** so every cell shows its output (answers, sources, the UI).
2. Save the notebook and upload `nestle_hr_assistant.ipynb`.

## Notes

* `gpt-3.5-turbo` is used because the brief asks for it; `gpt-4o-mini` is a cheaper, stronger drop-in replacement.
* The `.env` file and the `chroma_db/` index are git-ignored — never commit API keys.
