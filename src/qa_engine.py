import streamlit as st
from sentrnce_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

#Model loaders
@st.cache_resource
def load_embedding_model():
    # all-MiniLM-L6-v2 is the industry std lightweight embedding model
    # it encodes text into 384 dimensional vectors
    return SentenceTransformer('all-MiniLM-L6-v2')
@st.cache_resource
def load_qa_model():
    # flan-t5-base is a generative model — it WRITES answers
    # "text2text-generation" means: take text in, produce text out
    return pipeline('text2text-generation', model='google/flan-t5-base',device=-1)
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
def answer_question(question:str,chunks:list,chunk_embeddings)->dict:
    qa_model=load_qa_model()
    #Step 1: find semantically relevant chunks
    relevant_chunks= find_relevant_chunks(question,chunks,chunk_embeddings)
    #Step 2: build the prompt
    #flan-t5 needs a clear ins format
    context=" ".join(relevant_chunks)
    prompt=f"""Answer the following question based on the provided context.
If the answer is not in the context, say 'This information is not mentioned in the paper.'
Context: {context}
Question: {question}
Answer:"""
    # Step 3: generate the answer
    result = qa_model(
        prompt,
        max_new_tokens=200,    # max length of the generated answer
        do_sample=False        # deterministic output
    )
    answer = result[0]["generated_text"].strip()
    return {
        "answer": answer,
        "context_used": context[:300] + "..."   # show user which part of paper was used
    }