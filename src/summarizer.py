import streamlit as st
from transformers import pipeline
@st.cache_resource
def load_summarizer():
    #pipeline():Hugging Fcae's high level api that handles tokenizing,running the model, decoding o/p
    # device=-1 => means use CPU
    return pipeline(
        "text-generation",
        model='facebook/bart-large-cnn',
        device=-1
    ) 
def summarize_chunk(summarizer,text:str)->str:
    #skip chunks that are too short to summarize
    if len(text.split()) < 50:
        return text
    result=summarizer(text,max_length=150,min_length=40,do_sample=False)
    #result is a list of dicts with 'generated_text' as key
    return result[0]['generated_text']
def summarize_document(chunks:list)->str:
    #Step 1: Summarize each chunk
    summarizer=load_summarizer()
    chunk_summaries=[]
    for i,chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1} of {len(chunks)}...")
        chunk_summaries.append(summarize_chunk(summarizer, chunk))
    #Step 2: join all chunk summaries
    combined=' '.join(chunk_summaries)
    #Step 3: if combined is still long, do final pass
    #heirarchial summarization
    if len(combined.split()) > 300:
        print("Performing final summarization pass...")
        combined=summarize_chunk(summarizer,combined)
    return combined