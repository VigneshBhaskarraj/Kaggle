"""
Nestlé HR Assistant — standalone Gradio app.

This is the same Retrieval-Augmented Generation (RAG) pipeline that is walked through step by step in
`nestle_hr_assistant.ipynb`, packaged as a script so the chatbot can be run or deployed (e.g. on a server or
Hugging Face Spaces) without a notebook:

    python app.py            # local UI at http://127.0.0.1:7860
    python app.py --share    # additionally creates a temporary public Gradio link

Pipeline:  PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> OpenAI embeddings -> Chroma
           question -> top-k chunks -> prompt template -> GPT-3.5 Turbo -> answer (+ page citations)
"""
from __future__ import annotations

import argparse
import getpass
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Chroma phones home with anonymous usage stats by default; keep the app self-contained.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# langchain-community is being sunset in favour of standalone packages, but it is still where PyPDFLoader (required
# by the brief) lives and the pinned version works fine — silence its import-time notice.
warnings.filterwarnings("ignore", message=".*langchain-community.*")

import gradio as gr
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------------- configuration
PDF_PATH = Path("data/nestle_hr_policy.pdf")
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
FALLBACK_LLM_MODEL = "gpt-4o-mini"  # used if LLM_MODEL has been retired by OpenAI
COLLECTION_NAME = f"nestle_hr_policy__{EMBEDDING_MODEL}"  # one collection per embedding model -> never re-use stale vectors
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

NOT_FOUND_MESSAGE = "I couldn't find that in the Nestlé HR Policy document."

SYSTEM_PROMPT = f"""You are Nestlé's HR Assistant, a helpful and precise chatbot for employees and HR staff.
You answer questions using ONLY the excerpts provided from "The Nestlé Human Resources Policy".

Guidelines:
1. Ground every statement in the provided context. Never invent policies, numbers or entitlements.
2. If the context does not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}" and suggest contacting the local HR team.
3. Be concise and well structured — short paragraphs or bullet points.
4. When helpful, name the policy area the answer comes from (e.g. "Training and learning", "Total rewards").
5. Use the conversation history to understand follow-up questions, but still answer from the context.
6. Keep a professional, friendly tone."""


# ----------------------------------------------------------------------------- document processing
def load_pages(pdf_path: Path):
    """Load the PDF; returns one LangChain Document per page (with `page` metadata)."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path.resolve()} — see README for how to obtain it.")
    return PyPDFLoader(str(pdf_path)).load()


def split_pages(pages, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Split pages into overlapping chunks that fit comfortably into the prompt."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def build_vectorstore(chunks, embeddings, persist_dir: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
    """Embed the chunks into a persistent Chroma collection, re-using it if it is already complete."""
    existing = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_dir)
    n_existing = len(existing.get()["ids"])
    if n_existing == len(chunks):
        return existing
    if n_existing:
        existing.delete_collection()  # stale index (e.g. chunking changed) -> rebuild
    return Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=collection_name, persist_directory=persist_dir
    )


# ----------------------------------------------------------------------------- question answering
def format_docs(docs) -> str:
    """Render retrieved chunks as numbered, page-tagged context blocks."""
    return "\n\n".join(
        f"[Excerpt {i} — page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for i, d in enumerate(docs, start=1)
    )


def build_rag_chain(retriever, llm):
    """LCEL chain: {question, chat_history} -> {question, chat_history, docs, answer}."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "Context from the HR policy:\n{context}\n\nQuestion: {question}"),
        ]
    )
    retrieve = RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["question"]))
    generate = RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"])) | prompt | llm | StrOutputParser()
    return retrieve | RunnablePassthrough.assign(answer=generate)


def resolve_llm_model(model: str = LLM_MODEL) -> str:
    """Return `model` if this API key can use it, otherwise fall back (e.g. when a GPT-3.5 snapshot is retired)."""
    from openai import OpenAI

    try:
        OpenAI().models.retrieve(model)  # cheap metadata call — no tokens used
        return model
    except Exception as exc:
        print(f"'{model}' is not available for this API key ({type(exc).__name__}); using '{FALLBACK_LLM_MODEL}' instead")
        return FALLBACK_LLM_MODEL


def build_pipeline(
    pdf_path: Path = PDF_PATH,
    *,
    embeddings=None,
    llm=None,
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    top_k: int = TOP_K,
):
    """Wire the whole pipeline together. `embeddings`/`llm` can be injected (e.g. fakes for offline tests)."""
    embeddings = embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = llm or ChatOpenAI(model=resolve_llm_model(), temperature=0)
    chunks = split_pages(load_pages(pdf_path))
    vectorstore = build_vectorstore(chunks, embeddings, persist_dir, collection_name)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    return build_rag_chain(retriever, llm)


# ----------------------------------------------------------------------------- Gradio UI
def history_to_messages(history, max_turns: int = 6):
    """Convert Gradio chat history into LangChain messages (keeps the last `max_turns` exchanges)."""
    messages = []
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


def make_respond(chain):
    def respond(message, history):
        result = chain.invoke({"question": message, "chat_history": history_to_messages(history)})
        answer = result["answer"]
        if NOT_FOUND_MESSAGE.lower() not in answer.lower():
            pages_used = ", ".join(str(p) for p in sorted({d.metadata.get("page", 0) + 1 for d in result["docs"]}))
            answer += f"\n\n📄 *Source: The Nestlé Human Resources Policy — page(s) {pages_used}*"
        return answer

    return respond


def make_demo(chain) -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=make_respond(chain),
        title="Nestlé HR Assistant",
        description=(
            "Ask anything about **The Nestlé Human Resources Policy**. "
            "Answers are generated by GPT-3.5 Turbo from the policy text and cite the pages used."
        ),
        examples=[
            "What are the key principles of Nestlé's HR policy?",
            "How does Nestlé support employee training and development?",
            "What does the policy say about diversity and inclusion?",
            "How is employee performance evaluated and rewarded?",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Run the Nestlé HR Assistant chatbot.")
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    chain = build_pipeline()
    make_demo(chain).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
