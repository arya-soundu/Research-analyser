import os
import json
import hashlib

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_pdf_hash(file_bytes: bytes) -> str:
    """Compute the SHA-256 hash of a file's bytes."""
    return hashlib.sha256(file_bytes).hexdigest()

def save_to_cache(pdf_hash: str, filename: str, data: dict, pdf_bytes: bytes = None):
    """Save the paper data structure and its PDF bytes to local cache."""
    ensure_cache_dir()
    cache_path = os.path.join(CACHE_DIR, f"{pdf_hash}.json")
    
    cache_payload = {
        "filename": filename,
        "pdf_hash": pdf_hash,
        "payload": data
    }
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, ensure_ascii=False, indent=2)
        
    if pdf_bytes:
        pdf_path = os.path.join(CACHE_DIR, f"{pdf_hash}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

def load_from_cache(pdf_hash: str) -> dict:
    """Load cached paper data and raw PDF bytes if they exist."""
    cache_path = os.path.join(CACHE_DIR, f"{pdf_hash}.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        pdf_path = os.path.join(CACHE_DIR, f"{pdf_hash}.pdf")
        pdf_bytes = None
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
                
        data["pdf_bytes"] = pdf_bytes
        return data
    except Exception:
        return None

def list_recent_papers() -> list:
    """Return a list of recently analyzed papers sorted by modified time."""
    ensure_cache_dir()
    papers = []
    
    for filename in os.listdir(CACHE_DIR):
        if filename.endswith(".json"):
            cache_path = os.path.join(CACHE_DIR, filename)
            try:
                mtime = os.path.getmtime(cache_path)
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    payload = data.get("payload", {})
                    meta = payload.get("meta", {})
                    papers.append({
                        "hash": data.get("pdf_hash"),
                        "filename": data.get("filename"),
                        "title": meta.get("title", "Unknown Title"),
                        "mtime": mtime
                    })
            except Exception:
                continue
                
    # Sort descending by last modified time
    papers.sort(key=lambda x: x["mtime"], reverse=True)
    return papers
