import io
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pdf_extractor import extract_text_from_pdf, get_pdf_metadata
from src.preprocessor import clean_text,chunk_text,get_text_stats
from src.summarizer import load_summarizer,summarize_chunk
from src.structurer import build_structured_notes
from src.qa_engine import load_embedding_model, answer_question, embed_chunks
#UI - Page Config
st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Header
st.title("📄 Research Paper Analyzer")
st.markdown("Upload a PDF and get structured notes, summaries, and key insights.")
st.divider()

#Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    #slider = lets controll chunk size
    chunk_size=st.slider(
        label="Words per chunk",
        min_value=200,
        max_value=800,
        value=400,
        step=50
    )
    st.markdown("---")
    st.markdown("### 🧠 Architecture Stack")
    st.markdown("- **Summarization**: `bart-large-cnn` (Local)")
    st.markdown("- **Embeddings**: `all-MiniLM-L6-v2` (Local)")
    st.markdown("- **Generative Q&A**: `Llama-3.1-8B` (Groq API)")
    st.markdown("---")

    # pre-load the model while user reads the UI
    with st.spinner("Loading AI models..."):
        model = load_summarizer()
        load_embedding_model()
    st.success("✅ Semantic models ready")
    
    st.markdown("### 🔑 API Authentication")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "paste_your_key_here":
        st.warning("⚠️ Please add your `GROQ_API_KEY` to the hidden `.env` file to enable the Chat feature.")
    else:
        st.success("✅ Secure `.env` API Key loaded")

#---File Uploader--
#returns file object when user uploads, None otherwise
uploaded_file=st.file_uploader(
    label="Upload a PDF file",
    type=["pdf"]
)

#Main Logic
if uploaded_file:
    # Important: Reset session state if the user uploads a completely different file!
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.clear()
        st.session_state.current_file = uploaded_file.name

    file_bytes=uploaded_file.getvalue()
    #Step 1 - metadata
    meta=get_pdf_metadata(io.BytesIO(file_bytes))
    col1, col2, col3 = st.columns(3)
    col1.metric("Title",  meta["title"])
    col2.metric("Author", meta["author"])
    col3.metric("Pages",  meta["pages"])
    st.divider()

    #Step 2 & 3 - Extract & Preprocess
    if "chunks" not in st.session_state or st.session_state.get("last_chunk_size") != chunk_size:
        with st.spinner("Extracting and Preparing text..."):
            raw_text=extract_text_from_pdf(io.BytesIO(file_bytes))
            if raw_text.startswith("ERROR"):
                st.error(raw_text)
                st.stop()
            
            clean=clean_text(raw_text)
            st.session_state.chunks = chunk_text(clean,max_words=chunk_size)
            st.session_state.stats = get_text_stats(clean)
            st.session_state.last_chunk_size = chunk_size
            
            # If chunks change, blow away old dependencies
            for key in ["chunk_embeddings", "combined_summary", "structured_notes"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
    chunks = st.session_state.chunks
    stats = st.session_state.stats
    st.info(
        f"📊 **{stats['words_count']:,} words** · "
        f"**{stats['estimated_read_min']} min read** · "
        f"**{len(chunks)} chunks** to process")
        # pre-compute chunk embeddings once so Q&A is fast
    # stored in session_state so they survive Streamlit reruns
    if "chunk_embeddings" not in st.session_state:
        with st.spinner("🔢 Computing embeddings..."):
            st.session_state.chunk_embeddings = embed_chunks(chunks)


    #Step 4 - Summarize
    st.subheader("🤖 Generating Summary")
    
    if "combined_summary" not in st.session_state:
        progress=st.progress(0,text="Starting...")
        summaries=[]
        for i,chunk in enumerate(chunks):
            progress.progress(
                (i + 1) / len(chunks),
                text=f"Processing chunk {i+1} of {len(chunks)}..."
            )
            summaries.append(summarize_chunk(model,chunk))
        progress.empty() #remove when done
        st.session_state.combined_summary = " ".join(summaries)
        
    combined = st.session_state.combined_summary
    
    # ── STEP 5: structure and display ─────────────────────────
    if "structured_notes" not in st.session_state:
        with st.spinner("📂 Structuring notes..."):
            st.session_state.structured_notes = build_structured_notes(combined)
    
    notes = st.session_state.structured_notes
    st.success("✅ Analysis complete!")
    st.divider()
    # display results in tabs — keeps UI clean
    tab1, tab2, tab3 = st.tabs(["🗂️ Structured Notes", "📄 Full Summary","💬 Ask a Question"])
    with tab1:
        st.markdown("### 🎯 Objective")
        st.write(notes["objective"])
        st.markdown("---")
        st.markdown("### 🔬 Methodology")
        st.write(notes["methodology"])
        st.markdown("---")
        st.markdown("### 📊 Findings")
        st.write(notes["findings"])
        st.markdown("---")
        st.markdown("### ✅ Conclusion")
        st.write(notes["conclusion"])
    with tab2:
        st.write(combined)
    with tab3:
        st.markdown("Ask anything about the paper and the AI will find the answer.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # chat input pinned at bottom
        user_question = st.chat_input("Ask a question about the paper...")

        if user_question:
            with st.spinner("Thinking..."):
                result = answer_question(
                    user_question,
                    chunks,
                    st.session_state.chunk_embeddings,
                    api_key=api_key
                )
            
            # Insert the newest question at the TOP of the stack (index 0)
            st.session_state.chat_history.insert(0, {
                "question": user_question,
                "answer":   result["answer"],
                "context":  result["context_used"]
            })

        # Render the stacked history (Newest questions are at the top)
        for item in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                with st.expander("📄 Context used to answer"):
                    st.caption(item.get("context", "Context unavailable for this query."))

# ── EMPTY STATE ───────────────────────────────────────────────
# shown when no file uploaded yet
else:
    st.info("👆 Upload a PDF above to get started.")