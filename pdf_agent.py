"""
PDF Q&A Agent
=============
Chat with your PDF files. LlamaIndex splits the PDFs into chunks and indexes
them with free local embeddings, and a Claude-powered agent decides which
documents to search, reads the relevant passages, and answers with page citations.

Pipeline:
    PDFs -> pages -> chunks -> embeddings -> vector index (saved in ./storage)
    question -> agent -> [tools: list_documents, search_documents] -> cited answer
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.agent.workflow import FunctionAgent, ToolCall
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

load_dotenv()

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # free, runs locally
DATA_DIR = Path(os.getenv("PDF_DIR", "data"))
STORAGE_DIR = Path("storage")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the user's PDF documents.

Rules:
- ALWAYS use the search_documents tool before answering a question about the documents.
- You may search several times with different wording, or per document, to compare documents.
- Answer ONLY from the retrieved passages. If the answer isn't there, say you couldn't find it.
- Cite every fact with its source in the form (file_name, p. X).
- Be concise and well structured. Use bullet points for lists.
"""


# ---------------------------------------------------------------------------
# 1. Indexing: load PDFs -> split into chunks -> embed -> vector index
# ---------------------------------------------------------------------------
def configure_models(model: str | None = None) -> None:
    """Set the global LLM, embedding model and chunking strategy for LlamaIndex."""
    Settings.llm = Anthropic(model=model or DEFAULT_MODEL, temperature=0, max_tokens=2048)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


def _fingerprint(pdf_dir: Path) -> str:
    """A hash of file names, sizes and modification times, used to detect changed PDFs."""
    items = sorted((p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in pdf_dir.glob("*.pdf"))
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def load_or_build_index(pdf_dir: Path = DATA_DIR, rebuild: bool = False) -> VectorStoreIndex:
    """Load the saved index if the PDFs haven't changed; otherwise (re)build it."""
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in '{pdf_dir}/'. Add some PDFs and try again.")

    fingerprint = _fingerprint(pdf_dir)
    marker = STORAGE_DIR / "fingerprint.txt"
    if not rebuild and marker.exists() and marker.read_text() == fingerprint:
        storage = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        return load_index_from_storage(storage)

    # The PDF reader creates one Document per page, with metadata such as
    # file_name and page_label. That metadata is what makes page citations possible.
    documents = SimpleDirectoryReader(input_dir=str(pdf_dir), required_exts=[".pdf"]).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    marker.write_text(fingerprint)
    return index


# ---------------------------------------------------------------------------
# 2. Tools: plain Python functions the agent can decide to call
# ---------------------------------------------------------------------------
def make_tools(index: VectorStoreIndex, pdf_dir: Path, top_k: int = 5) -> list[FunctionTool]:
    def list_documents() -> str:
        """List the PDF documents that are available to search, with their page counts."""
        lines = []
        for path in sorted(pdf_dir.glob("*.pdf")):
            pages = {
                n.metadata.get("page_label")
                for n in index.docstore.docs.values()
                if n.metadata.get("file_name") == path.name
            }
            lines.append(f"- {path.name} ({len(pages)} pages)")
        return "\n".join(lines) or "No documents."

    def search_documents(query: str, file_name: Optional[str] = None) -> str:
        """Search the PDF documents for passages relevant to `query`.

        Args:
            query: What to look for. Use specific keywords from the question.
            file_name: Optional. Restrict the search to one PDF (exact name from list_documents).
        """
        filters = None
        if file_name:
            filters = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=file_name)])
        retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
        results = retriever.retrieve(query)
        if not results:
            return "No relevant passages found."
        passages = []
        for r in results:
            meta = r.node.metadata
            passages.append(
                f"[{meta.get('file_name', '?')}, p. {meta.get('page_label', '?')}] "
                f"(relevance {r.score:.2f})\n{r.node.get_content().strip()}"
            )
        return "\n\n---\n\n".join(passages)

    return [
        FunctionTool.from_defaults(fn=list_documents),
        FunctionTool.from_defaults(fn=search_documents),
    ]


# ---------------------------------------------------------------------------
# 3. The agent: Claude + tools + conversation memory
# ---------------------------------------------------------------------------
class PDFAgent:
    """A small synchronous wrapper around LlamaIndex's async FunctionAgent."""

    def __init__(self, pdf_dir: Path | str = DATA_DIR, model: str | None = None, rebuild: bool = False):
        configure_models(model)
        self.pdf_dir = Path(pdf_dir)
        self.index = load_or_build_index(self.pdf_dir, rebuild=rebuild)
        self.agent = FunctionAgent(
            tools=make_tools(self.index, self.pdf_dir),
            llm=Settings.llm,
            system_prompt=SYSTEM_PROMPT,
        )
        self._loop = asyncio.new_event_loop()
        self.reset()

    def reset(self) -> None:
        """Forget the conversation (start a new chat)."""
        # The memory buffer keeps the chat history between questions (for follow-ups).
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=40_000)

    async def _ask(self, question: str, on_tool_call: Callable[[str, dict], None] | None) -> str:
        handler = self.agent.run(user_msg=question, memory=self.memory)
        async for event in handler.stream_events():
            if isinstance(event, ToolCall) and on_tool_call:
                on_tool_call(event.tool_name, event.tool_kwargs)
        response = await handler
        return str(response)

    def ask(self, question: str, on_tool_call: Callable[[str, dict], None] | None = None) -> str:
        """Ask a question. `on_tool_call(name, args)` is called each time the agent uses a tool."""
        return self._loop.run_until_complete(self._ask(question, on_tool_call))
