import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_summarizer():
    # Keeping the original full-size BART model per your request
    model_name = 'facebook/bart-large-cnn'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return {"tokenizer": tokenizer, "model": model}

def summarize_chunk(model_pack: dict, text: str) -> str:
    # skip chunks that are too short
    if len(text.split()) < 50:
        return text
        
    tokenizer = model_pack["tokenizer"]
    model = model_pack["model"]
    
    # Tokenize input, ensure we truncate to BART's context length
    inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True)
    
    # Generate summary with greedy search (faster) to minimize CPU load
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=150, 
        min_length=40, 
        length_penalty=2.0
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)