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
def find_relevant_chunks(question:str,chunks:list,chunk_embeddings,top_k:int=3)->list:
    #Step 1:load the embedding model
    embed_model=load_embedding_model()
    #Step 2:convert the question into a vector 
    question_embedding=embed_model.encode(question,convert_to_tensor=True)
    #Step 3:compute cosine similarity between question vector and chunk vectors 
    scores = util.cos_sim(question_embedding,chunk_embeddings)[0]
    #Step 4:get top k chunks with highest scores indices
    top_indices = torch.topk(scores,k=min(top_k, len(chunks))).indices
    #Step 5:reutrn the actual chunk texts for those indices
    return [chunks[i] for i in top_indices]
def embed_chunks(chunks:list):
    #pre-compute embeddings for all chunks once paper is uploaded
    #storing means we dont have to recompute every time question is asked 
    embed_model=load_embedding_model()
    return embed_model.encode(chunks,convert_to_tensor=True)

#ANSWER GENERATION
def answer_question(question:str, chunks:list, chunk_embeddings, api_key:str)->dict:
    # Step 1: find semantically relevant chunks (happens locally)
    relevant_chunks = find_relevant_chunks(question, chunks, chunk_embeddings)
    context = " ".join(relevant_chunks)
    
    # Step 2: Initialize Groq
    if not api_key:
        return {
            "answer": "⚠️ Please enter your Groq API Key in the sidebar to use the Q&A feature.",
            "context_used": "No API Key provided."
        }
        
    client = Groq(api_key=api_key)
    
    # Step 3: Build the system message (instructions) and User message (the task)
    system_prompt = """You are a helpful research assistant. Answer the user's question in detail using the provided context as your primary source.
If the context does not contain the answer, you may use your general knowledge, but you MUST start your outside answer with: '[External Knowledge]'."""

    user_payload = f"Context: {context}\n\nQuestion: {question}"

    # Step 4: Stream the payload to Groq's blazing fast LPUs
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ],
        model="llama-3.1-8b-instant",
    )
    
    # Step 5: Return the AI's answer
    answer = chat_completion.choices[0].message.content
    
    return {
        "answer": answer,
        "context_used": context[:300] + "..."   # still showing the user which part of paper was used
    }