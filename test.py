from src.summarizer import load_summarizer, summarize_chunk

print("\n--- LOADING MODEL ---")
model = load_summarizer()
print("Model loaded.")

print("\n--- SUMMARIZING FIRST CHUNK ---")
chunks = ["This is a test chunk that we are using to verify if the summarizer is working correctly. " * 10]
test_summary = summarize_chunk(model, chunks[0])
print(test_summary)