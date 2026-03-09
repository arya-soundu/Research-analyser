from nt import P_NOWAITO
import PyPDF2
import io
def extract_text_from_pdf(file_bytes)->str:
    #file_bytes is an BytesIO object - treats it like a file
    reader=PyPDF2.PdfReader(file_bytes)
    all_text=[]
    for page in reader.pages:
        text=page.extract_text()
        if text:
            all_text.append(text)
    full_text="\n".join(all_text)
    if not full_text.strip():
        return "ERROR: Could not extract any text. PDF may be scanned or image based.."
    return full_text

def get_pdf_metadata(file_bytes)->dict:
    reader=PyPDF2.PdfReader(file_bytes)
    meta=reader.metadata
    return{
        "title":meta.title if meta and meta.title else "Unknown",
        "author":meta.author if meta and meta.author else "Unknown",
        "pages":len(reader.pages)
    }