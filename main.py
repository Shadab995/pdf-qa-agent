"""Command-line chat for the PDF Q&A Agent.

Usage:
    python main.py                          # chat with the PDFs in ./data
    python main.py --pdf-dir ~/papers       # use another folder
    python main.py --rebuild                # force re-indexing
"""

import argparse

from pdf_agent import DATA_DIR, PDFAgent


def show_tool_call(name: str, args: dict) -> None:
    if name == "search_documents":
        where = f" in {args['file_name']}" if args.get("file_name") else ""
        print(f"   🔎 searching{where}: \"{args.get('query', '')}\"")
    else:
        print(f"   📚 {name}()")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your PDF documents.")
    parser.add_argument("--pdf-dir", default=str(DATA_DIR), help="Folder containing PDF files (default: data)")
    parser.add_argument("--model", default=None, help="Claude model (default: CLAUDE_MODEL env var)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index even if PDFs haven't changed")
    args = parser.parse_args()

    print("📄 Loading and indexing PDFs (the first run downloads a small embedding model)...")
    agent = PDFAgent(pdf_dir=args.pdf_dir, model=args.model, rebuild=args.rebuild)
    print("✅ Ready! Ask anything about your documents. Commands: 'reset' = new chat, 'exit' = quit.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question.lower() == "reset":
            agent.reset()
            print("🧹 Conversation cleared.")
            continue
        if question:
            answer = agent.ask(question, on_tool_call=show_tool_call)
            print(f"\n🤖 {answer}")


if __name__ == "__main__":
    main()
