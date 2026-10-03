"""
Nestlé HR Assistant — an evidence-driven RAG pipeline over "The Nestlé Human Resources Policy".

This module is the single source of truth for the pipeline. `nestle_hr_assistant.ipynb` walks through the same code
cell by cell with the narrative, `app.py` serves it as a Gradio app and `tests/test_offline.py` verifies it without
an OpenAI key.

Design — each failure mode of a naive RAG chatbot gets a measured response:
  wrong passage retrieved   -> section-aware chunking + hybrid (BM25 + vector) retrieval with rank fusion
  follow-ups lose context   -> query rewriting into a standalone question before retrieval
  hallucinated entitlements -> grounded prompt, per-sentence groundedness check, explicit not-found fallback
  no provenance             -> section/page citations with quoted evidence and a confidence badge
  quality unknown           -> gold question set, retrieval/answer metrics and ablations
"""

# %% [imports]
from __future__ import annotations

import getpass
import hashlib
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Chroma phones home with anonymous usage stats by default; keep the project self-contained.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# langchain-community is being sunset, but it is still where PyPDFLoader (required by the brief) lives.
warnings.filterwarnings("ignore", message=".*langchain-community.*")

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# %% [config]
PDF_PATH = Path("data/nestle_hr_policy.pdf")
PDF_SOURCE_URL = "https://www.nestle.com/sites/default/files/asset-library/documents/jobs/the_nestle_hr_policy_pdf_2012.pdf"
PDF_SHA256 = "e9897ab73af6174211b22ac319a38832882616f35b2c71a364281cc4596b4545"
DOC_TITLE = "The Nestlé Human Resources Policy"

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")  # required by the brief
FALLBACK_LLM_MODEL = "gpt-4o-mini"  # used only if LLM_MODEL has been retired by OpenAI

TOP_K = 4
DEFAULT_THRESHOLDS = (0.50, 0.35)  # cosine-similarity cut-offs for High / Medium confidence (re-calibrated in the eval)
NOT_FOUND_MESSAGE = "I couldn't find that in the Nestlé HR Policy document."

# %% [openai_env]
def ensure_api_key() -> None:
    """Load OPENAI_API_KEY from the environment or a .env file, or prompt for it securely."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


def resolve_llm_model(model: str = LLM_MODEL) -> str:
    """Return `model` if this API key can use it, otherwise fall back (e.g. when a GPT-3.5 snapshot is retired)."""
    from openai import OpenAI

    try:
        OpenAI().models.retrieve(model)  # cheap metadata call — no tokens used
        return model
    except Exception as exc:
        print(f"'{model}' is not available for this API key ({type(exc).__name__}); using '{FALLBACK_LLM_MODEL}' instead")
        return FALLBACK_LLM_MODEL


# %% [loading]
def load_pages(pdf_path: Path = PDF_PATH) -> list[Document]:
    """PyPDFLoader: one Document per page, with 0-based `page` metadata."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path.resolve()} — see README.md")
    return PyPDFLoader(str(pdf_path)).load()


def verify_pdf(pdf_path: Path = PDF_PATH, expected_sha256: str = PDF_SHA256) -> bool:
    """Document provenance: confirm the file is the published policy (warns, never fails)."""
    digest = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    ok = digest == expected_sha256
    print(("✓" if ok else "!") + f" sha256 {digest[:16]}… " + ("matches the published document" if ok else "does NOT match the expected checksum"))
    return ok


# %% [sections]
EXPECTED_SECTIONS = [
    "Document information",
    "Introduction",
    "A shared responsibility",
    "Joining Nestlé",
    "Total rewards",
    "Employment and working conditions",
    "Training and learning",
    "Talent, development and performance management",
    "Employee relations",
    "A flexible and dynamic organisation",
]


@dataclass
class Section:
    title: str
    text: str
    pages: set[int] = field(default_factory=set)
    related_policy: str | None = None


def _dehyphenate(text: str) -> str:
    """Re-join words the PDF split across lines ('inter-\\nnational' -> 'international')."""
    return re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", text)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _heading_for(tokens, index: int) -> str:
    """Headings are extracted *after* their body (sidebar layout) — except when a section opens a page."""

    def scan(step: int):
        j = index + step
        while 0 <= j < len(tokens) and tokens[j][0] == "policy":
            j += step
        return tokens[j][1] if 0 <= j < len(tokens) and tokens[j][0] == "heading" else None

    return scan(+1) or scan(-1) or "Untitled"


def parse_sections(pages: list[Document], doc_title: str = DOC_TITLE, front_matter_pages: int = 2) -> list[Section]:
    """Rebuild the policy's structure from the extracted text.

    Inspecting PyPDFLoader's output showed three layout artefacts: a running header on every page, section headings
    emitted out of reading order (recognisable by a leading space), and 'Corporate policy: X' cross-references.
    """
    tokens: list[tuple[str, str, int]] = []  # (kind, text, page number)
    for page in pages:
        page_no = page.metadata["page"] + 1
        if page_no <= front_matter_pages:
            continue
        lines = _dehyphenate(page.page_content).split("\n")
        i = 0
        while i < len(lines):
            raw, text = lines[i], lines[i].strip()
            if not text:
                i += 1
                continue
            if text == doc_title and i + 1 < len(lines) and lines[i + 1].strip().isdigit():
                i += 2  # running header + page number
                continue
            if text.startswith("Corporate policy:"):
                name = text.split(":", 1)[1].strip()
                if not name and i + 1 < len(lines):
                    i += 1
                    name = lines[i].strip()
                tokens.append(("policy", name, page_no))
                i += 1
                continue
            is_heading = raw.startswith(" ") and 3 <= len(text) <= 60 and not text.endswith(".")
            tokens.append(("heading" if is_heading else "body", text, page_no))
            i += 1

    merged: list[tuple[str, str, set[int]]] = []
    for kind, text, page_no in tokens:
        continues_previous = bool(merged) and merged[-1][0] == kind and (
            kind == "body" or (kind == "heading" and page_no in merged[-1][2] and text[:1].islower())
        )  # a body spread over lines/pages, or a two-line heading ("Talent, development" / "and performance management")
        if continues_previous:
            prev = merged[-1]
            merged[-1] = (kind, f"{prev[1]} {text}", prev[2] | {page_no})
        else:
            merged.append((kind, text, {page_no}))

    sections: dict[str, Section] = {}
    for idx, (kind, text, page_nos) in enumerate(merged):
        if kind != "body":
            continue
        title = _heading_for(merged, idx)
        policy = merged[idx + 1][1] if idx + 1 < len(merged) and merged[idx + 1][0] == "policy" else None
        section = sections.setdefault(title, Section(title=title, text=""))
        section.text = f"{section.text} {text}".strip()
        section.pages |= page_nos
        section.related_policy = section.related_policy or policy

    front = " ".join(_normalise(_dehyphenate(p.page_content)) for p in pages[:front_matter_pages])
    info = Section("Document information", front, set(range(1, front_matter_pages + 1)))
    return [info, *sections.values()]


# %% [chunking]
def chunk_naive(pages, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """Baseline: split the raw page text, exactly as a first RAG tutorial would."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        page_no = chunk.metadata["page"] + 1
        chunk.metadata = {"chunk_id": i, "strategy": "naive", "section": "n/a", "page": page_no, "pages": str(page_no), "related_policy": ""}
    return chunks


def chunk_by_section(sections, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """Section-aware: same size as the baseline, but split *within* each section and prefix chunks with the title."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=[". ", " ", ""], keep_separator="end"
    )
    chunks: list[Document] = []
    for section in sections:
        body = section.text + (f" Related corporate policy: {section.related_policy}." if section.related_policy else "")
        for piece in splitter.split_text(body):
            chunks.append(
                Document(
                    page_content=f"[{section.title}]\n{piece.strip()}",
                    metadata={
                        "chunk_id": len(chunks),
                        "strategy": "section",
                        "section": section.title,
                        "page": min(section.pages),
                        "pages": ",".join(str(p) for p in sorted(section.pages)),
                        "related_policy": section.related_policy or "",
                    },
                )
            )
    return chunks


def make_chunks(strategy: str, pages, sections=None) -> list[Document]:
    if strategy == "naive":
        return chunk_naive(pages)
    if strategy == "section":
        return chunk_by_section(sections if sections is not None else parse_sections(pages))
    raise ValueError(f"unknown chunking strategy {strategy!r} (use 'naive' or 'section')")


# %% [vectorstore]
def collection_name_for(strategy: str, embedding_name: str = EMBEDDING_MODEL) -> str:
    """One Chroma collection per (chunking strategy, embedding model) -> stale vectors are never re-used."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", f"nestle_hr__{strategy}__{embedding_name}")


def build_vectorstore(chunks, embeddings, collection_name: str, persist_dir: str = CHROMA_DIR) -> Chroma:
    """Embed chunks into a persisted Chroma collection (cosine space); re-used when it is already complete."""
    kwargs = dict(collection_name=collection_name, persist_directory=persist_dir, collection_metadata={"hnsw:space": "cosine"})
    existing = Chroma(embedding_function=embeddings, **kwargs)
    n_existing = len(existing.get()["ids"])
    if n_existing == len(chunks):
        return existing
    if n_existing:
        existing.delete_collection()  # stale or partial index -> rebuild
    return Chroma.from_documents(documents=chunks, embedding=embeddings, **kwargs)


# %% [retrieval]
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever(BaseRetriever):
    """Keyword (BM25), semantic (Chroma) or hybrid retrieval, fused with Reciprocal Rank Fusion.

    Returned documents carry `score` (fused rank score) and `similarity` (cosine similarity from the vector ranker,
    None in keyword-only mode) in their metadata, so the trust layer can show confidence without a second search.
    """

    documents: list[Document]
    mode: str = "hybrid"  # "bm25" | "vector" | "hybrid"
    k: int = TOP_K
    candidate_k: int = 10  # candidates taken from each ranker before fusion
    rrf_k: int = 60  # RRF constant (Cormack et al., 2009)
    vectorstore: Any = None
    bm25: Any = None

    def _ranked_vector(self, query: str) -> list[tuple[int, float]]:
        hits = self.vectorstore.similarity_search_with_score(query, k=self.candidate_k)
        return [(doc.metadata["chunk_id"], 1.0 - distance) for doc, distance in hits]  # cosine distance -> similarity

    def _ranked_bm25(self, query: str) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.candidate_k]
        return [(i, float(scores[i])) for i in order]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        rankings, similarity = [], {}
        if self.mode in ("vector", "hybrid"):
            vector = self._ranked_vector(query)
            rankings.append(vector)
            similarity = dict(vector)
        if self.mode in ("bm25", "hybrid"):
            rankings.append(self._ranked_bm25(query))
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (chunk_id, _) in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        top = sorted(fused, key=lambda cid: -fused[cid])[: self.k]
        return [
            Document(
                page_content=self.documents[cid].page_content,
                metadata={**self.documents[cid].metadata, "score": round(fused[cid], 4), "similarity": similarity.get(cid)},
            )
            for cid in top
        ]


def make_retriever(
    chunks,
    *,
    mode: str = "hybrid",
    k: int = TOP_K,
    embeddings=None,
    collection_name: str | None = None,
    persist_dir: str = CHROMA_DIR,
    vectorstore=None,
) -> HybridRetriever:
    if mode not in ("bm25", "vector", "hybrid"):
        raise ValueError(f"unknown retrieval mode {mode!r} (use 'bm25', 'vector' or 'hybrid')")
    if mode != "bm25" and vectorstore is None:
        embeddings = embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL)
        name = collection_name or collection_name_for(chunks[0].metadata["strategy"])
        vectorstore = build_vectorstore(chunks, embeddings, name, persist_dir)
    bm25 = BM25Okapi([_tokenize(c.page_content) for c in chunks]) if mode != "vector" else None
    return HybridRetriever(documents=chunks, mode=mode, k=k, vectorstore=vectorstore if mode != "bm25" else None, bm25=bm25)


# %% [prompt_chain]
SYSTEM_PROMPT = f"""You are Nestlé's HR Assistant, a helpful and precise chatbot for employees and HR staff.
You answer questions using ONLY the excerpts provided from "The Nestlé Human Resources Policy".

Guidelines:
1. Ground every statement in the provided context. Never invent policies, numbers or entitlements.
2. If the context does not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}" and suggest contacting the local HR team.
3. Be concise and well structured — short paragraphs or bullet points.
4. Name the policy section the answer comes from when helpful (each excerpt starts with its section in brackets).
5. Use the conversation history to understand follow-up questions, but still answer from the context.
6. Keep a professional, friendly tone."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "Context from the HR policy:\n{context}\n\nQuestion: {question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest message as a single, self-contained question about Nestlé's HR policy, "
            "resolving references such as 'that', 'them' or 'it' from the conversation. Keep the user's intent and "
            "wording where possible. Return only the rewritten question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


def format_docs(docs) -> str:
    """Render retrieved chunks as numbered, section- and page-tagged excerpts."""
    return "\n\n".join(
        f"[Excerpt {i} — {d.metadata.get('section', 'n/a')}, page {d.metadata.get('pages', d.metadata.get('page', '?'))}]\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def build_rewriter(llm):
    return REWRITE_PROMPT | llm | StrOutputParser()


def build_rag_chain(retriever, llm, rewrite: bool = True):
    """LCEL chain: {question, chat_history} -> {question, chat_history, standalone_question, docs, answer}."""
    rewriter = build_rewriter(llm)

    def standalone(inputs: dict) -> str:
        if rewrite and inputs["chat_history"]:
            rewritten = rewriter.invoke({"question": inputs["question"], "chat_history": inputs["chat_history"]}).strip()
            return rewritten or inputs["question"]
        return inputs["question"]

    generate = RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"])) | ANSWER_PROMPT | llm | StrOutputParser()
    return (
        RunnablePassthrough.assign(chat_history=lambda x: x.get("chat_history") or [])
        | RunnablePassthrough.assign(standalone_question=RunnableLambda(standalone))
        | RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["standalone_question"]))
        | RunnablePassthrough.assign(answer=generate)
    )


def is_not_found(answer: str) -> bool:
    text = answer.lower()
    return NOT_FOUND_MESSAGE.lower() in text or "couldn't find that" in text or "could not find that" in text


# %% [trust]
STOPWORDS = set(
    "a an and are as at be but by can does for from has have how if in into is it its of on or that the their "
    "this to was we were what when where which who will with you your".split()
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) > 2 and t not in STOPWORDS}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 15]


def groundedness(answer: str, docs) -> dict:
    """Per-sentence lexical support: the share of a sentence's content words that occur in the retrieved context."""
    context = _content_tokens(" ".join(d.page_content for d in docs))
    sentences = _sentences(answer)
    grounded, flagged = 0, []
    for sentence in sentences:
        tokens = _content_tokens(sentence)
        support = len(tokens & context) / len(tokens) if tokens else 1.0
        if support >= 0.6:
            grounded += 1
        else:
            flagged.append((sentence, round(support, 2)))
    return {"grounded": grounded, "total": len(sentences), "flagged": flagged}


def top_similarity(docs) -> float | None:
    sims = [d.metadata.get("similarity") for d in docs if d.metadata.get("similarity") is not None]
    return max(sims) if sims else None


def confidence_label(similarity: float | None, thresholds=DEFAULT_THRESHOLDS) -> tuple[str, str]:
    high, medium = thresholds
    if similarity is None:
        return "n/a (keyword-only retrieval)", "⚪"
    if similarity >= high:
        return "High", "🟢"
    if similarity >= medium:
        return "Medium", "🟡"
    return "Low", "🔴"


def evidence_snippets(docs, query: str, max_chars: int = 180) -> list[tuple[str, str, str]]:
    """For each retrieved chunk, the sentence that best overlaps the question/answer — quotable provenance."""
    query_tokens = _content_tokens(query)
    snippets = []
    for d in docs:
        body = re.sub(r"^\[[^\]]*\]\n", "", d.page_content)  # drop the "[Section]" prefix
        best = max(_sentences(body) or [body], key=lambda s: len(_content_tokens(s) & query_tokens))
        if len(best) > max_chars:
            best = best[: max_chars - 1].rsplit(" ", 1)[0] + "…"
        snippets.append((d.metadata.get("section", "n/a"), str(d.metadata.get("pages", d.metadata.get("page", "?"))), best))
    return snippets


def render_answer(result: dict, thresholds=DEFAULT_THRESHOLDS, max_evidence: int = 3) -> str:
    """Answer + trust layer (confidence badge, groundedness, quoted evidence) as Markdown for the chat UI."""
    answer, docs = result["answer"].strip(), result["docs"]
    if is_not_found(answer):
        return f"{answer}\n\n🔴 **Confidence:** Low — this topic does not appear to be covered by the policy document."
    similarity = top_similarity(docs)
    label, icon = confidence_label(similarity, thresholds)
    g = groundedness(answer, docs)
    header = f"{icon} **Confidence:** {label}"
    if similarity is not None:
        header += f" (top match similarity {similarity:.2f})"
    header += f" · **Grounded:** {g['grounded']}/{g['total']} sentences"
    lines = [answer, "", header, "", "**Evidence from the policy**"]
    for section, pages, snippet in evidence_snippets(docs, f"{result['question']} {answer}")[:max_evidence]:
        lines.append(f'- *{section}* (p. {pages}): "{snippet}"')
    return "\n".join(lines)


# %% [gold]
@dataclass(frozen=True)
class GoldItem:
    question: str
    evidence: str = ""  # phrase that must appear in a retrieved chunk -> retrieval ground truth (chunking-agnostic)
    section: str = ""  # section the answer lives in
    keywords: tuple[str, ...] = ()  # terms a correct answer should mention -> answer ground truth
    history: tuple[tuple[str, str], ...] = ()  # prior (question, answer) turns for follow-up items
    kind: str = "in_scope"  # in_scope | follow_up | out_of_scope


GOLD_SET = [
    GoldItem("When was the HR policy issued and who approved it?", "Executive Board, Nestlé S.A.", "Document information", ("2012", "executive board")),
    GoldItem("Who is the target audience of the HR policy?", "audience All employees", "Document information", ("all employees",)),
    GoldItem("Which two documents is the HR policy built on?", "Corporate Business Principles", "Introduction", ("management and leadership principles", "corporate business principles")),
    GoldItem("How can the spirit of the HR policy be summarised in one sentence?", "people at the centre of everything we do", "Introduction", ("people at the centre",)),
    GoldItem("Who has the prime responsibility for people matters at Nestlé?", "Line managers have the prime responsibility", "A shared responsibility", ("line manager",)),
    GoldItem("What are the three areas of the Nestlé HR structure?", "Centres of Expertise", "A shared responsibility", ("centres of expertise", "business partners", "employee services")),
    GoldItem("What is the mission of HR managers and their teams?", "provide professional guidance", "A shared responsibility", ("guidance", "line managers")),
    GoldItem("What is considered when deciding to employ a person?", "Only relevant skills and experience", "Joining Nestlé", ("skills", "experience", "principles")),
    GoldItem("Does Nestlé consider a candidate's nationality or religion when hiring?", "consideration will be given to a candidate", "Joining Nestlé", ("nationality", "religion")),
    GoldItem("Who makes the final decision to hire a candidate?", "remains in the hands of the responsible", "Joining Nestlé", ("responsible manager", "manager")),
    GoldItem("What are the key elements of Nestlé's Total Rewards?", "Fixed Pay, Variable Pay, Benefits", "Total rewards", ("fixed pay", "variable pay", "benefits", "personal growth", "work life")),
    GoldItem("Who proposes an employee's remuneration?", "propose the remuneration", "Total rewards", ("manager",)),
    GoldItem("Which corporate policy covers total rewards?", "Nestlé Total Rewards Policy", "Total rewards", ("total rewards policy",)),
    GoldItem("Does Nestlé offer flexible working conditions?", "flexible working conditions whenever possible", "Employment and working conditions", ("flexible", "whenever possible")),
    GoldItem("What is Nestlé's position on harassment and discrimination?", "harassment or discrimination", "Employment and working conditions", ("harassment", "discrimination", "tolerate")),
    GoldItem("Who must take ownership of safety and health in their area?", "personal ownership of safety and health", "Employment and working conditions", ("line management", "safety", "health")),
    GoldItem("What is the primary source of learning at Nestlé?", "on-the-job training", "Training and learning", ("experience", "on-the-job")),
    GoldItem("Where are the international leadership training courses held?", "Rive-Reine", "Training and learning", ("rive-reine",)),
    GoldItem("Should attending a training programme be seen as a reward?", "never be considered as a reward", "Training and learning", ("reward", "development")),
    GoldItem("Which tools give employees feedback on their performance?", "Progress and Development Guide", "Talent, development and performance management", ("performance evaluation", "progress and development guide", "360")),
    GoldItem("What are promotions based on?", "promotions are based on sustained", "Talent, development and performance management", ("sustained performance", "potential")),
    GoldItem("How does Nestlé support career progression for women?", "mentoring schemes", "Talent, development and performance management", ("flexible", "mentoring", "dual career")),
    GoldItem("Can employees take international assignments?", "work in different countries", "Talent, development and performance management", ("international", "different countries")),
    GoldItem("Does Nestlé recognise freedom of association and collective bargaining?", "freedom of association", "Employee relations", ("freedom of association", "collective bargaining")),
    GoldItem("What values is Nestlé's culture built on?", "trust, mutual respect", "Employee relations", ("trust", "respect", "dialogue")),
    GoldItem("What kind of organisational structure is Nestlé committed to?", "flat and flexible structures", "A flexible and dynamic organisation", ("flat", "flexible", "minimal levels")),
    GoldItem("What is Nestlé's attitude to risk-taking and mistakes?", "people to take risks", "A flexible and dynamic organisation", ("risk", "mistake", "learn")),
]

# HR-sounding but *not* covered by this document — the assistant must decline instead of guessing.
OUT_OF_SCOPE = [
    GoldItem("What is the capital of France?", kind="out_of_scope"),
    GoldItem("How many days of annual leave do Nestlé employees get per year?", kind="out_of_scope"),
    GoldItem("What is the maternity leave entitlement at Nestlé?", kind="out_of_scope"),
    GoldItem("What is the salary of a marketing manager at Nestlé?", kind="out_of_scope"),
    GoldItem("Write a Python function that sorts a list.", kind="out_of_scope"),
]

# Follow-ups that only make sense given the previous turn (the prior answers are fixtures taken from the policy).
FOLLOW_UPS = [
    GoldItem("Who is responsible for proposing them for each employee?", "propose the remuneration", "Total rewards", ("manager",),
             history=(("What are the key elements of Total Rewards?", "Fixed Pay, Variable Pay, Benefits, Personal Growth and Development, and Work Life Environment."),), kind="follow_up"),
    GoldItem("Which corporate policy governs that?", "Expatriation Policy", "Talent, development and performance management", ("expatriation",),
             history=(("Does Nestlé offer international assignments?", "Yes — employees interested in international assignments can be given the opportunity to work in different countries."),), kind="follow_up"),
    GoldItem("What do those courses aim to build?", "integrated business understanding", "Training and learning", ("business understanding", "values"),
             history=(("Where do leaders attend international training courses?", "At Rive-Reine, or in programmes run by Nestlé's strategic learning partners."),), kind="follow_up"),
    GoldItem("And what is HR's role in supporting them?", "provide professional guidance", "A shared responsibility", ("guidance",),
             history=(("Who has prime responsibility for people matters?", "Line managers."),), kind="follow_up"),
    GoldItem("Who is in charge of the employee's own development?", "charge of her or his own professional", "Talent, development and performance management", ("employee",),
             history=(("Which tools give employees performance feedback?", "The Performance Evaluation process (PE), the Progress and Development Guide (PDG) and 360° assessments."),), kind="follow_up"),
]


# %% [evaluation]
def contains_evidence(doc: Document, evidence: str) -> bool:
    return _normalise(evidence).lower() in _normalise(doc.page_content).lower()


def history_messages(item: GoldItem) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for question, answer in item.history:
        messages += [HumanMessage(question), AIMessage(answer)]
    return messages


def evidence_rank(docs, evidence: str) -> int | None:
    for rank, doc in enumerate(docs):
        if contains_evidence(doc, evidence):
            return rank
    return None


def retrieval_metrics(retriever, items, ks=(1, 4), queries=None) -> dict:
    """Hit@k and MRR over gold items; `queries` optionally overrides the text sent to the retriever."""
    queries = queries or [item.question for item in items]
    ranks = [evidence_rank(retriever.invoke(q), item.evidence) for q, item in zip(queries, items)]
    metrics = {f"Hit@{k}": sum(r is not None and r < k for r in ranks) / len(items) for k in ks}
    metrics["MRR"] = sum(1 / (r + 1) for r in ranks if r is not None) / len(items)
    return metrics


def run_retrieval_ablation(
    pages,
    sections,
    *,
    embeddings=None,
    embedding_name: str = EMBEDDING_MODEL,
    strategies=("naive", "section"),
    modes=("bm25", "vector", "hybrid"),
    items=None,
    k_max: int = 6,
    persist_dir: str = CHROMA_DIR,
):
    """Chunking strategy × retrieval mode grid on the gold set -> (DataFrame, {(strategy, mode): retriever})."""
    items = items or GOLD_SET
    rows, retrievers = [], {}
    for strategy in strategies:
        chunks = make_chunks(strategy, pages, sections)
        vectorstore = None
        if any(mode != "bm25" for mode in modes):
            embeddings = embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = build_vectorstore(chunks, embeddings, collection_name_for(strategy, embedding_name), persist_dir)
        for mode in modes:
            retriever = make_retriever(chunks, mode=mode, k=k_max, vectorstore=vectorstore)
            retrievers[(strategy, mode)] = retriever
            rows.append({"chunking": strategy, "retrieval": mode, "chunks": len(chunks), **retrieval_metrics(retriever, items)})
    return pd.DataFrame(rows), retrievers


PREFERENCE = {"section": 0, "naive": 1, "hybrid": 0, "vector": 1, "bm25": 2}


def choose_best(ablation: pd.DataFrame, n_items: int = len(GOLD_SET)) -> pd.Series:
    """Pick the configuration the chatbot ships with.

    Hit@4 first, then Hit@1, then MRR — but a difference smaller than one question (1 / n_items) is noise on a gold
    set this size, so such near-ties are resolved by design preference: section-aware + hybrid, which also keeps the
    similarity scores the confidence badge needs.
    """
    tolerance = 1.0 / n_items + 1e-9
    candidates = ablation[ablation["Hit@4"] >= ablation["Hit@4"].max() - tolerance]
    candidates = candidates[candidates["Hit@1"] >= candidates["Hit@1"].max() - tolerance]
    ranked = candidates.assign(_pref=candidates["chunking"].map(PREFERENCE) + candidates["retrieval"].map(PREFERENCE))
    ranked = ranked.sort_values(["_pref", "MRR"], ascending=[True, False])
    return ranked.iloc[0].drop("_pref")


def k_sweep(retriever, items=None, ks=(2, 4, 6)) -> pd.DataFrame:
    """How many chunks to pass to the model: Hit@k for several k from one retrieval per question."""
    items = items or GOLD_SET
    assert retriever.k >= max(ks), "retriever.k must be at least max(ks)"
    ranks = [evidence_rank(retriever.invoke(item.question), item.evidence) for item in items]
    return pd.DataFrame({"k": list(ks), "Hit@k": [sum(r is not None and r < k for r in ranks) / len(items) for k in ks]})


def rewrite_ablation(retriever, llm, items=None, k: int = TOP_K) -> pd.DataFrame:
    """Follow-up questions retrieved as typed vs. after rewriting them into standalone questions."""
    items = items or FOLLOW_UPS
    rewriter = build_rewriter(llm)
    rows = []
    for item in items:
        rewritten = rewriter.invoke({"question": item.question, "chat_history": history_messages(item)}).strip() or item.question
        as_typed = evidence_rank(retriever.invoke(item.question)[:k], item.evidence)
        after = evidence_rank(retriever.invoke(rewritten)[:k], item.evidence)
        rows.append({"follow-up": item.question, "rewritten": rewritten, f"hit@{k} as typed": as_typed is not None, f"hit@{k} rewritten": after is not None})
    return pd.DataFrame(rows)


def keyword_coverage(answer: str, keywords) -> float:
    text = answer.lower()
    return sum(kw.lower() in text for kw in keywords) / len(keywords) if keywords else float("nan")


def answer_metrics(chain, items=None, out_of_scope=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """End-to-end quality: keyword coverage + groundedness on in-scope questions, decline rate on out-of-scope ones."""
    items, out_of_scope = items or GOLD_SET, out_of_scope or OUT_OF_SCOPE
    rows = []
    for item in items:
        result = chain.invoke({"question": item.question, "chat_history": history_messages(item)})
        g = groundedness(result["answer"], result["docs"])
        rows.append(
            {
                "question": item.question,
                "section": item.section,
                "keyword coverage": keyword_coverage(result["answer"], item.keywords),
                "grounded": g["grounded"] / g["total"] if g["total"] else float("nan"),
                "top similarity": top_similarity(result["docs"]),
                "declined": is_not_found(result["answer"]),
                "answer": result["answer"],
            }
        )
    oos = []
    for item in out_of_scope:
        result = chain.invoke({"question": item.question, "chat_history": []})
        oos.append({"question": item.question, "declined": is_not_found(result["answer"]), "top similarity": top_similarity(result["docs"]), "answer": result["answer"]})
    df_in, df_out = pd.DataFrame(rows), pd.DataFrame(oos)
    summary = {
        "keyword coverage (mean)": float(df_in["keyword coverage"].mean()),
        "answers with coverage ≥ 0.5": float((df_in["keyword coverage"] >= 0.5).mean()),
        "grounded sentences (mean)": float(df_in["grounded"].mean()),
        "in-scope wrongly declined": float(df_in["declined"].mean()),
        "out-of-scope correctly declined": float(df_out["declined"].mean()),
    }
    return df_in, df_out, summary


def calibrate_thresholds(retriever, in_scope=None, out_of_scope=None, default=DEFAULT_THRESHOLDS):
    """Derive the High/Medium confidence cut-offs from the top-similarity distributions of in- vs out-of-scope questions."""
    in_scope, out_of_scope = in_scope or GOLD_SET, out_of_scope or OUT_OF_SCOPE
    sims_in = [top_similarity(retriever.invoke(item.question)) for item in in_scope]
    sims_out = [top_similarity(retriever.invoke(item.question)) for item in out_of_scope]
    if any(s is None for s in sims_in + sims_out):
        return default, pd.DataFrame()  # keyword-only retrieval has no similarity to calibrate on
    s_in, s_out = pd.Series(sims_in), pd.Series(sims_out)
    high = float(s_in.quantile(0.25))  # three quarters of genuine policy questions score at least this
    medium = float(min((s_out.median() + high) / 2, high - 0.01))  # halfway down to a typical off-topic question
    table = pd.DataFrame({"in-scope": s_in.describe(), "out-of-scope": s_out.describe()})
    return (round(high, 3), round(medium, 3)), table


def summarize_findings(ablation: pd.DataFrame, best: pd.Series, k_table=None, rewrite_table=None, answer_summary=None) -> list[str]:
    """Turn the measured numbers into plain-language findings (recomputed on every run, never hand-written)."""

    def row(chunking, retrieval):
        match = ablation[(ablation["chunking"] == chunking) & (ablation["retrieval"] == retrieval)]
        return match.iloc[0] if len(match) else None

    lines = [
        f"Best configuration: {best['chunking']} chunking + {best['retrieval']} retrieval "
        f"(Hit@4 = {best['Hit@4']:.2f}, Hit@1 = {best['Hit@1']:.2f}, MRR = {best['MRR']:.2f})."
    ]
    naive, section = row("naive", best["retrieval"]), row("section", best["retrieval"])
    if naive is not None and section is not None:
        lines.append(
            f"Chunking ({best['retrieval']} retrieval): naive → section-aware moves Hit@1 {naive['Hit@1']:.2f} → {section['Hit@1']:.2f} "
            f"and Hit@4 {naive['Hit@4']:.2f} → {section['Hit@4']:.2f}."
        )
    hybrid = row(best["chunking"], "hybrid")
    for other in ("vector", "bm25"):
        single = row(best["chunking"], other)
        if single is not None and hybrid is not None:
            lines.append(
                f"Retrieval ({best['chunking']} chunking): {other}-only Hit@1 {single['Hit@1']:.2f} / Hit@4 {single['Hit@4']:.2f} "
                f"vs hybrid {hybrid['Hit@1']:.2f} / {hybrid['Hit@4']:.2f}."
            )
    if k_table is not None and len(k_table):
        best_k = int(k_table.loc[k_table["Hit@k"] == k_table["Hit@k"].max(), "k"].min())
        sweep = ", ".join(f"k={int(r['k'])} → {r['Hit@k']:.2f}" for _, r in k_table.iterrows())
        lines.append(f"Context size: Hit@k {sweep}; the smallest k reaching the best hit rate is {best_k}.")
    if rewrite_table is not None and len(rewrite_table):
        cols = [c for c in rewrite_table.columns if c.startswith("hit@")]
        lines.append(f"Follow-ups: {cols[0]} {rewrite_table[cols[0]].mean():.2f} → {cols[1]} {rewrite_table[cols[1]].mean():.2f} with query rewriting.")
    if answer_summary:
        lines.append("Answers: " + "; ".join(f"{name} {value:.2f}" for name, value in answer_summary.items()) + ".")
    return lines


# %% [ui]
def history_to_messages(history, max_turns: int = 6) -> list[BaseMessage]:
    """Convert Gradio chat history into LangChain messages (keeps the last `max_turns` exchanges)."""
    messages: list[BaseMessage] = []
    for item in history:
        if isinstance(item, dict):  # Gradio "messages" format
            content = item.get("content")
            if isinstance(content, list):  # multimodal payloads -> keep the text parts
                content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            if item.get("role") == "user":
                messages.append(HumanMessage(str(content)))
            elif item.get("role") == "assistant":
                messages.append(AIMessage(str(content)))
        else:  # legacy (user, assistant) tuples
            user_msg, bot_msg = item
            messages += [HumanMessage(str(user_msg)), AIMessage(str(bot_msg or ""))]
    return messages[-max_turns * 2 :]


def make_respond(chain, thresholds=DEFAULT_THRESHOLDS):
    def respond(message, history):
        result = chain.invoke({"question": message, "chat_history": history_to_messages(history)})
        return render_answer(result, thresholds)

    return respond


def make_demo(chain, thresholds=DEFAULT_THRESHOLDS) -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=make_respond(chain, thresholds),
        title="Nestlé HR Assistant",
        description=(
            "Ask anything about **The Nestlé Human Resources Policy**. Answers are grounded in the policy text, "
            "cite the section and page they come from, and show how confident the assistant is."
        ),
        examples=[
            "What are the key elements of Nestlé's Total Rewards?",
            "How does Nestlé support employee training and development?",
            "Does Nestlé offer flexible working conditions?",
            "What are promotions based on?",
        ],
    )


# %% [pipeline]
def build_pipeline(
    pdf_path: Path = PDF_PATH,
    *,
    strategy: str = "section",
    mode: str = "hybrid",
    k: int = TOP_K,
    rewrite: bool = True,
    embeddings=None,
    llm=None,
    embedding_name: str = EMBEDDING_MODEL,
    persist_dir: str = CHROMA_DIR,
):
    """Wire everything together -> (rag_chain, retriever). `embeddings`/`llm` can be injected (e.g. fakes for tests)."""
    pages = load_pages(pdf_path)
    sections = parse_sections(pages)
    chunks = make_chunks(strategy, pages, sections)
    retriever = make_retriever(
        chunks, mode=mode, k=k, embeddings=embeddings, collection_name=collection_name_for(strategy, embedding_name), persist_dir=persist_dir
    )
    llm = llm or ChatOpenAI(model=resolve_llm_model(), temperature=0)
    return build_rag_chain(retriever, llm, rewrite=rewrite), retriever
