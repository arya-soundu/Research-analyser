import streamlit as st
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import os
import torch

#Model loaders
@st.cache_resource
def load_embedding_model():
    # all-MiniLM-L6-v2 is the industry std lightweight embedding model
    # it encodes text into 384 dimensional vectors
    return SentenceTransformer('all-MiniLM-L6-v2')
# Note: We deleted load_qa_model() completely! 
# We no longer load a huge conversational model into your laptop's RAM. 
# Instead, Groq's cloud LPU handles it using their REST API.
#Semantic Retrieval
def find_relevant_chunks(question:str,chunks:list,chunk_embeddings,top_k:int=3,original_question:str=None)->tuple:
    #Step 1:load the embedding model
    embed_model=load_embedding_model()
    #Step 2:convert the question into a vector 
    question_embedding=embed_model.encode(question,convert_to_tensor=True)
    #Step 3:compute cosine similarity between question vector and chunk vectors 
    scores = util.cos_sim(question_embedding,chunk_embeddings)[0]
    
    # Keyword Boosting: detect references to figures, tables, or sections in the question
    # and boost chunks containing matching terms to ensure precise retrieval.
    import re
    # Match various abbreviations (figure, fig, table, tbl, section, sec) followed by a number
    matches = re.findall(r'(figure|fig\.?|table|tbl\.?|section|sec\.?)\s*(\d+)', question.lower())
    if original_question:
        matches += re.findall(r'(figure|fig\.?|table|tbl\.?|section|sec\.?)\s*(\d+)', original_question.lower())
    matches = list(set(matches))
    
    for label, num in matches:
        search_terms = []
        if label.startswith("fig") or label.startswith("figure"):
            search_terms = [
                f"figure {num}", f"figure{num}",
                f"fig. {num}", f"fig.{num}",
                f"fig {num}", f"fig{num}"
            ]
        elif label.startswith("tab") or label.startswith("tbl"):
            search_terms = [
                f"table {num}", f"table{num}",
                f"tbl. {num}", f"tbl.{num}",
                f"tbl {num}", f"tbl{num}"
            ]
        elif label.startswith("sec") or label.startswith("section"):
            search_terms = [
                f"section {num}", f"section{num}",
                f"sec. {num}", f"sec.{num}",
                f"sec {num}", f"sec{num}"
            ]
            
        # Add score boost for matching chunks
        for idx, chunk in enumerate(chunks):
            # Normalize all Unicode whitespace/spacing characters to standard space for robust check
            chunk_normalized = re.sub(r'\s+', ' ', chunk.lower())
            if any(term in chunk_normalized for term in search_terms):
                scores[idx] += 1.0
                
    # Calculate the max score after boosting to evaluate document relevance
    max_boosted_score = torch.max(scores).item()
    
    #Step 4:get top k chunks with highest scores indices
    top_indices = torch.topk(scores,k=min(top_k, len(chunks))).indices.tolist()
    #Step 5:return the actual chunk texts, their indices, and the max boosted similarity score
    return [chunks[i] for i in top_indices], top_indices, max_boosted_score

def embed_chunks(chunks:list):
    #pre-compute embeddings for all chunks once paper is uploaded
    #storing means we dont have to recompute every time question is asked 
    embed_model=load_embedding_model()
    return embed_model.encode(chunks,convert_to_tensor=True)

#ANSWER GENERATION
def answer_question(question:str, chunks:list, chunk_embeddings, api_key:str, chat_history:list=None, chunk_pages:list=None)->dict:
    # Step 1: Initialize Groq
    if not api_key:
        return {
            "answer": "⚠️ Please enter your Groq API Key in the sidebar to use the Q&A feature.",
            "context_used": "No API Key provided.",
            "sources": []
        }
        
    client = Groq(api_key=api_key)
    
    # Step 2: Reformulate follow-up question if chat history exists
    query_for_retrieval = question
    if chat_history and len(chat_history) > 0:
        try:
            history_str = ""
            for turn in reversed(chat_history[:3]):
                history_str += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"
            
            reformulate_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question (for a semantic search vector search).
If it is already standalone or doesn't refer to previous turns, output it exactly as is.
Do not include any commentary, prefixes, or quotes. Output ONLY the rephrased question.

History:
{history_str}
Follow-up Question: {question}
Standalone Question:"""

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise search query generator. Output ONLY the rephrased search query, nothing else."},
                    {"role": "user", "content": reformulate_prompt}
                ],
                model="llama-3.1-8b-instant",
                max_tokens=60,
                temperature=0.0
            )
            rephrased = chat_completion.choices[0].message.content.strip()
            rephrased = rephrased.strip('"\'')
            if rephrased:
                query_for_retrieval = rephrased
        except Exception:
            pass # Fallback to original question

    # Step 3: Find semantically relevant chunks (happens locally) using reformulated query
    relevant_chunks, top_indices, max_boosted_score = find_relevant_chunks(query_for_retrieval, chunks, chunk_embeddings, original_question=question)
    context = " ".join(relevant_chunks)
    
    is_external_only = (max_boosted_score < 0.15)
    
    # Map indices to source page numbers and text contents
    retrieved_sources = []
    if chunk_pages and not is_external_only:
        for idx in top_indices:
            if idx < len(chunk_pages):
                page_num = chunk_pages[idx]
                chunk_text_content = chunks[idx]
                retrieved_sources.append({
                    "page": page_num,
                    "text": chunk_text_content
                })
    
    # Step 4: Build the system message (instructions) and messages timeline
    system_prompt = """You are a helpful research assistant. Answer the user's question in detail.
Use the provided context as your primary source of truth.
- If the context does not contain the answer, or if the context is insufficient or irrelevant, you MUST answer the question using your general knowledge (web knowledge).
- IMPORTANT: If you answer the question using your general/external knowledge (because the context is missing, insufficient, or irrelevant), you MUST prefix your response with the exact tag '[EXTERNAL_KNOWLEDGE]' at the very beginning of your response. Otherwise, do NOT include the tag.

Example response using context:
Based on the paper, the authors proposed...

Example response using general/external knowledge:
[EXTERNAL_KNOWLEDGE]
Based on general knowledge, Python's quicksort can be implemented as..."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Append recent chat history (oldest first)
    if chat_history:
        for turn in reversed(chat_history[:3]):
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
            
    # Append the current query with the retrieved context
    if is_external_only:
        user_content = f"Question: {question}\n(Note: This question appears to be outside the scope of the document. Please answer using your general/external knowledge and prefix the response with '[EXTERNAL_KNOWLEDGE]'.)"
    else:
        user_content = f"Context: {context}\n\nQuestion: {question}"
        
    messages.append({"role": "user", "content": user_content})

    # Step 5: Stream the payload to Groq's blazing fast LPUs
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.1-8b-instant",
    )
    
    # Step 6: Return the AI's answer
    answer = chat_completion.choices[0].message.content
    
    is_external = is_external_only or ("[EXTERNAL_KNOWLEDGE]" in answer)
    if "[EXTERNAL_KNOWLEDGE]" in answer:
        answer = answer.replace("[EXTERNAL_KNOWLEDGE]", "").strip()
        
    return {
        "answer": answer,
        "context_used": "No document context used (General Knowledge)." if is_external else context[:300] + "...",
        "sources": [] if is_external else retrieved_sources,
        "is_external": is_external
    }