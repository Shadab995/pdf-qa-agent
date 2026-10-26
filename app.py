"""Streamlit chat UI for the PDF Q&A Agent.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from pdf_agent import DATA_DIR, PDFAgent

st.set_page_config(page_title="PDF Q&A Agent", page_icon="📄", layout="wide")
st.title("📄 PDF Q&A Agent")
st.caption("LlamaIndex + Claude: chat with your PDFs and get answers with page citations.")

DATA_DIR.mkdir(exist_ok=True)

with st.sidebar:
    st.header("📚 Documents")
    uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    if uploads:
        for f in uploads:
            (DATA_DIR / Path(f.name).name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploads)} file(s). Click 'Re-index' to include them.")
    for pdf in sorted(Path(DATA_DIR).glob("*.pdf")):
        st.write(f"• {pdf.name}")
    if st.button("🔄 Re-index documents"):
        st.session_state.pop("agent", None)
        st.session_state.messages = []
    if st.button("🧹 New chat"):
        st.session_state.messages = []
        if "agent" in st.session_state:
            st.session_state.agent.reset()

if "agent" not in st.session_state:
    try:
        with st.spinner("Indexing PDFs (the first run downloads a small embedding model)..."):
            st.session_state.agent = PDFAgent()
    except FileNotFoundError as exc:
        st.info(f"{exc} Use the sidebar to upload PDFs.")
        st.stop()

st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Searching documents...", expanded=False) as status:
            def on_tool_call(name: str, args: dict) -> None:
                if name == "search_documents":
                    where = f" in *{args['file_name']}*" if args.get("file_name") else ""
                    st.write(f"🔎 Searched{where}: \"{args.get('query', '')}\"")
                else:
                    st.write(f"📚 {name}()")

            answer = st.session_state.agent.ask(question, on_tool_call=on_tool_call)
            status.update(label="Done", state="complete")
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
