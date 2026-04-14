import io
import streamlit as st

from src.pdf_extractor import extract_text_from_pdf, get_pdf_metadata
from src.preprocessor import clean_text,chunk_text,get_text_stats
from src.summarizer import load_summarizer,summarize_chunk
from src.structurer import build_structured_notes
from src.qa_engine import load_embedding_model, load_qa_model,answer_question,embed_chunks
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
    st.markdown("**Model:** `facebook/bart-large-cnn`")
    st.markdown("**Framework:** HuggingFace Transformers")
    st.markdown("---")

    # pre-load the model while user reads the UI
    # @st.cache_resource means this only runs ONCE per session
    with st.spinner("Loading AI models..."):
        model = load_summarizer()
        load_qa_model()
        load_embedding_model()
    st.success("✅ All models ready")

#---File Uploader--
#returns file object when user uploads, None otherwise
uploaded_file=st.file_uploader(
    label="Upload a PDF file",
    type=["pdf"]
)

#Main Logic
if uploaded_file:
    file_bytes=uploaded_file.getvalue()
    #Step 1 - metadata
    meta=get_pdf_metadata(io.BytesIO(file_bytes))
    col1, col2, col3 = st.columns(3)
    col1.metric("Title",  meta["title"])
    col2.metric("Author", meta["author"])
    col3.metric("Pages",  meta["pages"])
    st.divider()

    #Step 2 - Extract text
    with st.spinner("Extracting text..."):
        raw_text=extract_text_from_pdf(io.BytesIO(file_bytes))
    #if extarction failed show error and stop
    if raw_text.startswith("ERROR"):
        st.error(raw_text)
        st.stop()

    #Step 3 - Preprocess
    with st.spinner("Preparing text..."):
        clean=clean_text(raw_text)
        chunks=chunk_text(clean,max_words=chunk_size)
        stats=get_text_stats(clean)
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
    progress=st.progress(0,text="Starting...")
    summaries=[]
    for i,chunk in enumerate(chunks):
        progress.progress(
            (i + 1) / len(chunks),
            text=f"Processing chunk {i+1} of {len(chunks)}..."
        )
        summaries.append(summarize_chunk(model,chunk))
    progress.empty() #remove when done
    combined=" ".join(summaries)
    # We deliberately skip a "Final compression pass" here because forcing multiple pages 
    # of detailed summaries into a single 150-word box deletes 90% of the paper's findings!
    
    # ── STEP 5: structure and display ─────────────────────────
    with st.spinner("📂 Structuring notes..."):
        notes = build_structured_notes(combined)
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

        # display previous questions and answers
        for item in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])

        # chat input pinned at bottom
        user_question = st.chat_input("Ask a question about the paper...")

        if user_question:
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = answer_question(
                        user_question,
                        chunks,
                        st.session_state.chunk_embeddings
                    )
                st.write(result["answer"])
                with st.expander("📄 Context used to answer"):
                    st.caption(result["context_used"])

            st.session_state.chat_history.append({
                "question": user_question,
                "answer":   result["answer"]
            })

# ── EMPTY STATE ───────────────────────────────────────────────
# shown when no file uploaded yet
else:
    st.info("👆 Upload a PDF above to get started.")