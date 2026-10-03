"""
Offline end-to-end test of the RAG pipeline in app.py — no OpenAI key or network needed.

Real PDF loading, splitting and Chroma indexing are exercised; OpenAI embeddings and GPT-3.5 Turbo are replaced with
LangChain's deterministic fakes, so the wiring can be verified for free.

    python tests/test_offline.py        # or: python -m pytest tests/
"""
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OPENAI_API_KEY", "sk-offline-test")  # never used: fakes are injected below
warnings.filterwarnings("ignore")

import gradio as gr
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import app

PDF = REPO / "data" / "nestle_hr_policy.pdf"


def test_pipeline_offline():
    pages = app.load_pages(PDF)
    chunks = app.split_pages(pages)
    assert len(pages) >= 5 and len(chunks) > len(pages), "PDF should load into pages and split into more chunks"
    assert all(len(c.page_content) <= app.CHUNK_SIZE for c in chunks)
    assert all("page" in c.metadata for c in chunks)

    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        embeddings = DeterministicFakeEmbedding(size=64)
        llm = FakeListChatModel(responses=["Grounded fake answer.", app.NOT_FOUND_MESSAGE])
        chain = app.build_pipeline(PDF, embeddings=embeddings, llm=llm, persist_dir=tmp, collection_name="test")

        # the persisted collection is re-used instead of being rebuilt
        store = app.build_vectorstore(chunks, embeddings, tmp, "test")
        assert len(store.get()["ids"]) == len(chunks)

        result = chain.invoke({"question": "How does Nestlé support training?", "chat_history": []})
        assert result["answer"] == "Grounded fake answer."
        assert len(result["docs"]) == app.TOP_K

        respond = app.make_respond(chain)
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        not_found = respond("What is the capital of France?", history)
        assert app.NOT_FOUND_MESSAGE in not_found and "Source" not in not_found
        cited = respond("Training?", history)
        assert "Source: The Nestlé Human Resources Policy" in cited

        messages = app.history_to_messages(history + [("legacy question", "legacy answer")])
        assert [type(m).__name__ for m in messages] == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]

        assert isinstance(app.make_demo(chain), gr.ChatInterface)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_pipeline_offline()
    print("offline pipeline test passed")
