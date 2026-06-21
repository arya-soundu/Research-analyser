import io
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pdf_extractor import extract_text_from_pdf, get_pdf_metadata, extract_images_from_pdf, extract_pages_from_pdf, render_pdf_page_to_image
from src.preprocessor import clean_text,chunk_text,get_text_stats
from src.summarizer import load_summarizer,summarize_chunk
from src.structurer import build_structured_notes
from src.qa_engine import load_embedding_model, answer_question, embed_chunks
from src.cache_manager import get_pdf_hash, save_to_cache, load_from_cache, list_recent_papers

def load_paper_into_session(cached_data, current_chunk_size=400):
    # clear session state to prevent cross-contamination
    st.session_state.clear()
    
    payload = cached_data["payload"]
    st.session_state.current_file = cached_data["filename"]
    st.session_state.pdf_hash = cached_data["pdf_hash"]
    
    # Restore components
    st.session_state.meta = payload["meta"]
    st.session_state.chunks = payload["chunks"]
    st.session_state.chunk_pages = payload["chunk_pages"]
    st.session_state.stats = payload["stats"]
    st.session_state.combined_summary = payload["combined_summary"]
    st.session_state.structured_notes = payload["structured_notes"]
    st.session_state.chat_history = payload["chat_history"]
    
    # Set the chunk size variables to prevent triggering Step 2 & 3
    st.session_state.last_chunk_size = current_chunk_size
    
    # Restore PDF bytes if available
    if cached_data.get("pdf_bytes"):
        st.session_state.pdf_bytes = cached_data["pdf_bytes"]

def generate_citations(meta: dict) -> dict:
    title = meta.get("title", "Unknown Title")
    author = meta.get("author", "Unknown Author")
    pages = meta.get("pages", 1)
    
    author_clean = author.strip() if author else "Unknown Author"
    if ";" in author_clean:
        author_list = [a.strip() for a in author_clean.split(";")]
        if len(author_list) > 2:
            author_apa = f"{author_list[0]} et al."
            author_mla = f"{author_list[0]}, et al."
        elif len(author_list) == 2:
            author_apa = f"{author_list[0]} & {author_list[1]}"
            author_mla = f"{author_list[0]} and {author_list[1]}"
        else:
            author_apa = author_list[0]
            author_mla = author_list[0]
    else:
        author_apa = author_clean
        author_mla = author_clean
        
    year = "2026"
    
    apa = f"{author_apa}. ({year}). *{title}*."
    mla = f"{author_mla}. *\"{title}\"*."
    
    cite_key = "".join(c.lower() for c in title if c.isalnum() or c.isspace()).replace(" ", "_")[:15]
    if not cite_key:
        cite_key = "paper_citation"
        
    bibtex = f"""@article{{{cite_key},
  author    = {{{author_clean}}},
  title     = {{{title}}},
  year      = {{{year}}},
  pages     = {{1-{pages}}}
}}"""
    
    return {"apa": apa, "mla": mla, "bibtex": bibtex}

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

    st.markdown("### 🧠 AI Model Status")
    st.info("Models load automatically upon PDF upload.")
    
    st.markdown("### 🔑 API Authentication")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "paste_your_key_here":
        st.warning("⚠️ Please add your `GROQ_API_KEY` to the hidden `.env` file to enable the Chat feature.")
    else:
        st.success("✅ Secure `.env` API Key loaded")

    # Load recent papers selector
    recent_papers = list_recent_papers()
    if recent_papers:
        st.markdown("---")
        st.markdown("### 📂 Load Recent Paper")
        options = ["-- Select a paper --"] + [f"{p['title']} ({p['filename']})" for p in recent_papers]
        selected_option = st.selectbox("Select paper to restore:", options, index=0)
        if selected_option != "-- Select a paper --":
            opt_idx = options.index(selected_option) - 1
            paper_hash = recent_papers[opt_idx]["hash"]
            if st.session_state.get("pdf_hash") != paper_hash:
                cached = load_from_cache(paper_hash)
                if cached:
                    load_paper_into_session(cached, chunk_size)
                    st.rerun()

#---File Uploader--
#returns file object when user uploads, None otherwise
uploaded_file=st.file_uploader(
    label="Upload a PDF file",
    type=["pdf"]
)

#Main Logic
file_bytes = None
filename = None
pdf_hash = None

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    pdf_hash = get_pdf_hash(file_bytes)
    
    # Check if a different file is uploaded
    if st.session_state.get("pdf_hash") != pdf_hash:
        cached = load_from_cache(pdf_hash)
        if cached:
            load_paper_into_session(cached, chunk_size)
            # Make sure we store the uploaded bytes
            st.session_state.pdf_bytes = file_bytes
            st.rerun()
        else:
            # Clean start for a new file
            st.session_state.clear()
            st.session_state.current_file = filename
            st.session_state.pdf_hash = pdf_hash
            st.session_state.pdf_bytes = file_bytes
elif st.session_state.get("current_file"):
    file_bytes = st.session_state.get("pdf_bytes")
    filename = st.session_state.get("current_file")
    pdf_hash = st.session_state.get("pdf_hash")

if file_bytes:
    # Load AI models (will load instantly from cache after the first download)
    with st.spinner("Loading local AI models (first run downloads ~1.7 GB)..."):
        model = load_summarizer()
        load_embedding_model()

    # Setup columns for side-by-side view if a page is selected
    selected_page = st.session_state.get("selected_page")
    if selected_page:
        col_main, col_viewer = st.columns([1.2, 0.8])
    else:
        col_main = st.container()
        col_viewer = None

    with col_main:
        #Step 1 - metadata
        if "meta" not in st.session_state:
            st.session_state.meta = get_pdf_metadata(file_bytes)
        meta = st.session_state.meta
        
        with st.sidebar:
            st.markdown("---")
            with st.expander("🎓 Academic Citations"):
                cites = generate_citations(meta)
                st.markdown("**APA Style**")
                st.code(cites["apa"], language="markdown")
                st.markdown("**MLA Style**")
                st.code(cites["mla"], language="markdown")
                st.markdown("**BibTeX Format**")
                st.code(cites["bibtex"], language="bibtex")

        col1, col2, col3 = st.columns(3)
        col1.metric("Title",  meta["title"])
        col2.metric("Author", meta["author"])
        col3.metric("Pages",  meta["pages"])
        st.divider()

        #Step 2 & 3 - Extract & Preprocess
        if "chunks" not in st.session_state or st.session_state.get("last_chunk_size") != chunk_size:
            with st.spinner("Extracting multi-modal text and imagery..."):
                pages_text = extract_pages_from_pdf(file_bytes)
                if not pages_text or not any(p.strip() for p in pages_text):
                    st.error("ERROR: Could not extract any text. PDF may be scanned or image based.")
                    st.stop()
                
                # Instantly slice out all visual data
                st.session_state.extracted_images = extract_images_from_pdf(file_bytes)
                
                chunks = []
                chunk_pages = []
                for page_idx, page_text in enumerate(pages_text):
                    page_num = page_idx + 1
                    clean = clean_text(page_text)
                    if not clean.strip():
                        continue
                    page_chunks = chunk_text(clean, max_words=chunk_size)
                    for chunk in page_chunks:
                        chunks.append(chunk)
                        chunk_pages.append(page_num)
                
                st.session_state.chunks = chunks
                st.session_state.chunk_pages = chunk_pages
                
                full_clean_text = "\n".join(pages_text)
                st.session_state.stats = get_text_stats(full_clean_text)
                st.session_state.last_chunk_size = chunk_size
                
                # If chunks change, blow away old dependencies
                for key in ["chunk_embeddings", "combined_summary", "structured_notes"]:
                    if key in st.session_state:
                        del st.session_state[key]
                        
        chunks = st.session_state.chunks
        chunk_pages = st.session_state.chunk_pages
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
                
                # Save initial analysis state to local cache folder
                cache_data = {
                    "meta": meta,
                    "chunks": chunks,
                    "chunk_pages": chunk_pages,
                    "stats": stats,
                    "combined_summary": combined,
                    "structured_notes": st.session_state.structured_notes,
                    "chat_history": []
                }
                save_to_cache(pdf_hash, filename, cache_data, file_bytes)
        
        notes = st.session_state.structured_notes
        st.success("✅ Analysis complete!")
        st.divider()
        # display results in tabs — keeps UI clean
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🗂️ Structured Notes", "🖼️ Visual Findings", "📄 Full Summary", "💬 Ask a Question"],
            key="main_tabs"
        )
        
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
            st.markdown("---")
            
            # Export Markdown button
            doc_title = meta["title"] if meta["title"] != "Unknown" else uploaded_file.name
            markdown_content = f"""# 📄 Research Summary: {doc_title}

**Author:** {meta["author"]}
**Pages:** {meta["pages"]}

---

## 🎯 Objective
{notes["objective"]}

## 🔬 Methodology
{notes["methodology"]}

## 📊 Findings
{notes["findings"]}

## ✅ Conclusion
{notes["conclusion"]}

---

## 📄 Full Summary
{combined}
"""
            safe_filename = "".join(c for c in doc_title if c.isalnum() or c.isspace()).replace(" ", "_")[:45]
            st.download_button(
                label="📥 Export Summary to Markdown (Obsidian/Notion)",
                data=markdown_content,
                file_name=f"{safe_filename}_summary.md",
                mime="text/markdown",
                key="download_summary_md"
            )
            
        with tab2:
            st.markdown("### 🖼️ Graphical Highlights")
            if "extracted_images" not in st.session_state or not st.session_state.extracted_images:
                if file_bytes:
                    with st.spinner("Extracting visual highlights from document..."):
                        st.session_state.extracted_images = extract_images_from_pdf(file_bytes)
                else:
                    st.session_state.extracted_images = []
            
            images_list = st.session_state.get("extracted_images", [])
            if not images_list:
                st.info("No explicit embedded images or charts were found in this document.")
            else:
                st.markdown(f"*{len(images_list)} visual figures extracted from paper:*")
                # Create a simple grid layout for the images
                for i in range(0, len(images_list), 2):
                    cols = st.columns(2)
                    with cols[0]:
                        st.image(images_list[i], use_container_width=True, caption=f"Figure {i+1}")
                    if i + 1 < len(images_list):
                        with cols[1]:
                            st.image(images_list[i+1], use_container_width=True, caption=f"Figure {i+2}")
                            
        with tab3:
            st.write(combined)
        with tab4:
            col_chat_header, col_clear_btn = st.columns([3, 1])
            with col_chat_header:
                st.markdown("Ask anything about the paper and the AI will find the answer.")
            with col_clear_btn:
                if st.button("🧹 Clear Chat", key="clear_chat_history_btn"):
                    st.session_state.chat_history = []
                    # Update local cache with cleared history
                    cache_data = {
                        "meta": meta,
                        "chunks": chunks,
                        "chunk_pages": chunk_pages,
                        "stats": stats,
                        "combined_summary": combined,
                        "structured_notes": notes,
                        "chat_history": []
                    }
                    save_to_cache(pdf_hash, filename, cache_data, file_bytes)
                    st.rerun()

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
                        api_key=api_key,
                        chat_history=st.session_state.chat_history,
                        chunk_pages=st.session_state.chunk_pages
                    )
                
                # Insert the newest question at the TOP of the stack (index 0)
                st.session_state.chat_history.insert(0, {
                    "question": user_question,
                    "answer":   result["answer"],
                    "context":  result["context_used"],
                    "sources":  result.get("sources", [])
                })
                
                # Save updated chat history to local cache file
                cache_data = {
                    "meta": meta,
                    "chunks": chunks,
                    "chunk_pages": chunk_pages,
                    "stats": stats,
                    "combined_summary": combined,
                    "structured_notes": notes,
                    "chat_history": st.session_state.chat_history
                }
                save_to_cache(pdf_hash, filename, cache_data, file_bytes)

            # Render the stacked history (Newest questions are at the top)
            for item_idx, item in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(item["question"])
                with st.chat_message("assistant"):
                    st.write(item["answer"])
                    
                    # Render source citations
                    sources = item.get("sources", [])
                    if sources:
                        # Extract unique pages (handles both old list of ints and new list of dicts)
                        unique_pages = []
                        seen = set()
                        for s in sources:
                            p = s["page"] if isinstance(s, dict) else s
                            if p not in seen:
                                seen.add(p)
                                unique_pages.append(p)
                        
                        cols = st.columns(len(unique_pages) + 2)
                        cols[0].markdown("**Citations:**")
                        for s_idx, page_num in enumerate(unique_pages):
                            btn_key = f"cite_{item_idx}_{page_num}_{s_idx}"
                            if cols[s_idx + 1].button(f"📄 Page {page_num}", key=btn_key):
                                st.session_state.selected_page = page_num
                                # Extract matching chunks for this page in this turn (if dict)
                                match_texts = [s["text"] for s in sources if isinstance(s, dict) and s["page"] == page_num]
                                st.session_state.highlight_context = match_texts
                                st.rerun()
                    
                    with st.expander("📄 Context used to answer"):
                        st.caption(item.get("context", "Context unavailable for this query."))

    if col_viewer and selected_page:
        with col_viewer:
            st.subheader(f"📄 PDF Page Viewer - Page {selected_page}")
            if st.button("Close Viewer ✖️", key="close_pdf_viewer"):
                st.session_state.selected_page = None
                st.session_state.highlight_context = None
                st.rerun()
            
            with st.spinner("Rendering PDF page..."):
                @st.cache_data
                def get_cached_page_image(bytes_data, p_num, hl_texts_tuple):
                    hl_list = list(hl_texts_tuple) if hl_texts_tuple else None
                    return render_pdf_page_to_image(bytes_data, p_num, hl_list)
                
                try:
                    hl_context = st.session_state.get("highlight_context", [])
                    hl_tuple = tuple(hl_context) if hl_context else None
                    page_img = get_cached_page_image(file_bytes, selected_page, hl_tuple)
                    st.image(page_img, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to render page: {str(e)}")

# ── EMPTY STATE ───────────────────────────────────────────────
# shown when no file uploaded yet
else:
    st.info("👆 Upload a PDF above to get started.")