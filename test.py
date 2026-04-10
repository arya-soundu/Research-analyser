from src.structurer import build_structured_notes

# use a fake summary to test classification logic
test_summary = """This study investigates the application of InLegalBERT for summarizing Indian legal documents. 
We propose a framework that uses K-Means clustering combined with sentence embeddings. 
The model achieved a ROUGE-L score of 0.3858 at 30% compression ratio, outperforming BART and T5. 
In conclusion, domain-specific models show clear advantages for legal text summarization tasks."""

notes = build_structured_notes(test_summary)

print("\n--- STRUCTURED NOTES ---")
for section, content in notes.items():
    print(f"\n{section.upper()}:")
    print(content)