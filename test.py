import io
from src.pdf_extractor import extract_text_from_pdf,get_pdf_metadata
from src.preprocessor import clean_text, chunk_text, get_text_stats

with open(r"C:\Users\sriso\OneDrive\Desktop\CS\Mini Project\LegalTextSummarization_InLegalBERT.pdf","rb") as file:
    file_bytes=file.read()
    text=extract_text_from_pdf(io.BytesIO(file_bytes))
    print("Text:",text[:500])
    metadata=get_pdf_metadata(io.BytesIO(file_bytes))
    print("Metadata:",metadata)

clean = clean_text(text)
chunks = chunk_text(clean)
stats = get_text_stats(clean)

print("--- STATS ---")
print(stats)

print(f"\n--- NUMBER OF CHUNKS ---")
print(len(chunks))

print("\n--- FIRST CHUNK PREVIEW ---")
print(chunks[0][:300])