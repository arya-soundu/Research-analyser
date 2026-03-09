import re

def clean_text(raw_text)->str:
    #Fix 1- hyphenated words broken across lines.. \w matches any word characters
    text=re.sub(r'(\w)-\n(\w)',r'\1\2',raw_text)
    #Fix 2- replace all newlines with a space
    text=re.sub(r'\n', ' ', text)
    #Fix 3- collapse multiple spaces into single space
    text=re.sub(r' +', ' ',text)
    # Fix 4- remove journal header lines like "Vol:.(1234567890)Int J Syst..."
    text = re.sub(r'Vol:\..*?(?=\s)', '', text)
    # Fix 5- remove DOI lines — not useful for summarization
    text = re.sub(r'https://doi\.org\S+', '', text)
    #Fix 6- remove received/revised/accepted lines
    text=re.sub(r'Received:.*?(?=\w{4,})', '', text)
    #Fix 7- remove copyright line
    text = re.sub(r'©.*?(?=\w{4,})', '', text)
    return text.strip()

def chunk_text(text: str, max_words: int = 400) -> list:
    # NEW STRATEGY: split by sentences, not paragraphs
    # This works reliably on two-column PDFs where paragraph
    sentences = re.split(r'(?<=[.!?]) (?=[A-Z])', text)
    chunks = []
    current_chunk = []
    current_count = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        # skip very short "sentences" — these are usually noise like
        # page numbers, figure labels, table headers
        if word_count < 5:
            continue
        if current_count + word_count > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_count = 0
        current_chunk.append(sentence)
        current_count += word_count
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def get_text_stats(text:str)->dict:
    words=text.split()
    return {
        "words_count":len(words),
        "estimated_read_min":round(len(words)/200)
    }
