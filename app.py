"""
Nestlé HR Assistant — standalone Gradio app.

The pipeline lives in `hr_assistant.py` (and is walked through step by step in `nestle_hr_assistant.ipynb`).
This script serves it as a chat application:

    python app.py                       # http://127.0.0.1:7860, section-aware chunking + hybrid retrieval
    python app.py --share               # additionally creates a temporary public Gradio link
    python app.py --strategy naive --mode vector --k 6 --no-rewrite    # run any configuration from the ablation
"""
from __future__ import annotations

import argparse

from hr_assistant import TOP_K, build_pipeline, ensure_api_key, make_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Nestlé HR Assistant chatbot.")
    parser.add_argument("--strategy", choices=["naive", "section"], default="section", help="chunking strategy")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid", help="retrieval mode")
    parser.add_argument("--k", type=int, default=TOP_K, help="chunks passed to the model per question")
    parser.add_argument("--no-rewrite", action="store_true", help="disable follow-up query rewriting")
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    ensure_api_key()
    chain, _ = build_pipeline(strategy=args.strategy, mode=args.mode, k=args.k, rewrite=not args.no_rewrite)
    make_demo(chain).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
