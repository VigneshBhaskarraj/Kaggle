"""
Offline tests for the pipeline in hr_assistant.py — no OpenAI key or network needed.

PDF loading, section parsing, chunking, BM25 retrieval and Chroma indexing are exercised for real; OpenAI embeddings
and GPT-3.5 Turbo are replaced with LangChain's deterministic fakes so the wiring and the evaluation harness can be
verified for free. The BM25 test is a genuine retrieval-quality check against the gold question set.

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
import pandas as pd
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import hr_assistant as hr

PDF = REPO / "data" / "nestle_hr_policy.pdf"
PAGES = hr.load_pages(PDF)
SECTIONS = hr.parse_sections(PAGES)


def fake_llm(*responses):
    return FakeListChatModel(responses=list(responses) or ["Grounded fake answer about the policy."])


def test_sections_are_recovered():
    assert [s.title for s in SECTIONS] == hr.EXPECTED_SECTIONS
    by_title = {s.title: s for s in SECTIONS}
    assert by_title["Total rewards"].related_policy == "Nestlé Total Rewards Policy"
    assert by_title["Talent, development and performance management"].related_policy == "Expatriation Policy"
    assert by_title["Joining Nestlé"].pages == {4} and by_title["Total rewards"].pages == {5}
    assert "international assignments" in by_title["Talent, development and performance management"].text  # de-hyphenated


def test_chunking_and_gold_set_consistency():
    for strategy in ("naive", "section"):
        chunks = hr.make_chunks(strategy, PAGES, SECTIONS)
        assert len(chunks) > len(PAGES)
        assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
        for item in hr.GOLD_SET + hr.FOLLOW_UPS:
            holders = [c for c in chunks if hr.contains_evidence(c, item.evidence)]
            assert holders, f"{strategy}: evidence not found in any chunk: {item.evidence!r}"
            if strategy == "section":
                assert item.section in {c.metadata["section"] for c in holders}, f"{item.evidence!r} not in section {item.section!r}"


def test_bm25_retrieval_quality():  # real retrieval signal, no API needed
    retriever = hr.make_retriever(hr.make_chunks("section", PAGES, SECTIONS), mode="bm25", k=6)
    metrics = hr.retrieval_metrics(retriever, hr.GOLD_SET)
    assert metrics["Hit@4"] >= 0.85, metrics
    assert metrics["Hit@1"] >= 0.70, metrics
    docs = retriever.invoke("Where are the international leadership training courses held?")
    assert docs[0].metadata["section"] == "Training and learning" and docs[0].metadata["similarity"] is None


def test_pipeline_with_fakes():
    tmp = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        embeddings = DeterministicFakeEmbedding(size=64)
        chain, retriever = hr.build_pipeline(
            PDF, strategy="section", mode="hybrid", k=4, embeddings=embeddings,
            llm=fake_llm("Fixed Pay, Variable Pay, Benefits, Personal Growth and Development and Work Life Environment.",
                         "Who proposes the remuneration of each employee?",  # rewrite call
                         "Each manager proposes the remuneration of their employees.",
                         hr.NOT_FOUND_MESSAGE + " Please contact your local HR team."),
            embedding_name="fake", persist_dir=tmp,
        )
        store = hr.build_vectorstore(retriever.documents, embeddings, hr.collection_name_for("section", "fake"), tmp)
        assert len(store.get()["ids"]) == len(retriever.documents)  # persisted collection is re-used, not rebuilt

        result = chain.invoke({"question": "What are the key elements of Total Rewards?"})
        assert set(result) >= {"question", "chat_history", "standalone_question", "docs", "answer"}
        assert result["standalone_question"] == result["question"] and len(result["docs"]) == 4
        assert all("similarity" in d.metadata and "score" in d.metadata for d in result["docs"])
        rendered = hr.render_answer(result)
        assert "**Confidence:**" in rendered and "**Grounded:**" in rendered and "Evidence from the policy" in rendered

        history = [{"role": "user", "content": "What are the key elements of Total Rewards?"}, {"role": "assistant", "content": result["answer"]}]
        follow_up = chain.invoke({"question": "Who proposes them?", "chat_history": hr.history_to_messages(history)})
        assert follow_up["standalone_question"] == "Who proposes the remuneration of each employee?"  # rewriting happened

        declined = hr.make_respond(chain)("What is the capital of France?", [])  # no history -> no rewrite call
        assert hr.NOT_FOUND_MESSAGE in declined and "Evidence" not in declined and "Low" in declined

        messages = hr.history_to_messages(history + [("legacy question", "legacy answer")])
        assert [type(m).__name__ for m in messages] == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
        assert isinstance(hr.make_demo(chain), gr.ChatInterface)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluation_harness_with_fakes():
    tmp = tempfile.mkdtemp(prefix="chroma_eval_")
    try:
        embeddings = DeterministicFakeEmbedding(size=64)
        ablation, retrievers = hr.run_retrieval_ablation(PAGES, SECTIONS, embeddings=embeddings, embedding_name="fake", persist_dir=tmp)
        assert len(ablation) == 6 and {"Hit@1", "Hit@4", "MRR", "chunks"} <= set(ablation.columns)
        assert ablation.loc[(ablation.chunking == "section") & (ablation.retrieval == "bm25"), "Hit@4"].item() >= 0.85
        best = hr.choose_best(ablation)
        assert (best["chunking"], best["retrieval"]) in retrievers

        retriever = retrievers[("section", "hybrid")]
        k_table = hr.k_sweep(retriever)
        assert list(k_table["k"]) == [2, 4, 6] and k_table["Hit@k"].is_monotonic_increasing

        rewrites = hr.rewrite_ablation(retriever, fake_llm("Who proposes the remuneration of each employee?"))
        assert len(rewrites) == len(hr.FOLLOW_UPS) and "rewritten" in rewrites.columns

        chain = hr.build_rag_chain(retriever, fake_llm("Line managers have the prime responsibility.", hr.NOT_FOUND_MESSAGE), rewrite=False)
        df_in, df_out, summary = hr.answer_metrics(chain, hr.GOLD_SET[:4], hr.OUT_OF_SCOPE[:2])
        assert len(df_in) == 4 and len(df_out) == 2 and 0.0 <= summary["out-of-scope correctly declined"] <= 1.0

        thresholds, table = hr.calibrate_thresholds(retriever)
        assert thresholds[0] > thresholds[1] and isinstance(table, pd.DataFrame)
        findings = hr.summarize_findings(ablation, best, k_table, rewrites, summary)
        assert len(findings) >= 5 and all(isinstance(line, str) for line in findings)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"ok  {name}")
    print("all offline tests passed")
