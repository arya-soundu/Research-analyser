import fitz  # PyMuPDF
from PIL import Image
import io

def sort_blocks_geometrically(blocks, page_width) -> str:
    # Filter text blocks and ignore empty ones (block_type 0 is text)
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return ""
        
    midpoint = page_width / 2
    
    # Sort all blocks top-to-bottom first
    sorted_by_y = sorted(text_blocks, key=lambda b: b[1])
    
    segments = []
    current_left = []
    current_right = []
    
    for b in sorted_by_y:
        x0, y0, x1, y1, text, block_no, block_type = b
        
        # Check if the block is spanning across the midpoint (full-width element like title/header/footer)
        is_full_width = (x0 < midpoint - 30) and (x1 > midpoint + 30) and (x1 - x0 > 0.5 * page_width)
        
        if is_full_width:
            # Flush accumulated column blocks in reading order (left column, then right column)
            if current_left or current_right:
                segments.extend(sorted(current_left, key=lambda x: x[1]))
                segments.extend(sorted(current_right, key=lambda x: x[1]))
                current_left = []
                current_right = []
            segments.append(b)
        else:
            # Classify as left or right column
            if x1 <= midpoint + 15:
                current_left.append(b)
            elif x0 >= midpoint - 15:
                current_right.append(b)
            else:
                # Fallback based on centroid
                centroid_x = (x0 + x1) / 2
                if centroid_x < midpoint:
                    current_left.append(b)
                else:
                    current_right.append(b)
                    
    # Flush remaining blocks
    if current_left or current_right:
        segments.extend(sorted(current_left, key=lambda x: x[1]))
        segments.extend(sorted(current_right, key=lambda x: x[1]))
        
    return "\n".join(b[4] for b in segments)

def extract_text_from_pdf(file_bytes_string: bytes) -> str:
    # fitz safely reads raw bytes via the stream parameter
    doc = fitz.open(stream=file_bytes_string, filetype="pdf")
    all_text = []
    
    for page in doc:
        # Extract blocks containing coordinate data
        blocks = page.get_text("blocks")
        page_width = page.rect.width
        text = sort_blocks_geometrically(blocks, page_width)
        if text:
            all_text.append(text)
            
    full_text = "\n".join(all_text)
    if not full_text.strip():
        return "ERROR: Could not extract any text. PDF may be scanned or image based."
    return full_text

def get_pdf_metadata(file_bytes_string: bytes) -> dict:
    doc = fitz.open(stream=file_bytes_string, filetype="pdf")
    meta = doc.metadata
    return {
        "title": meta.get("title", "Unknown") if meta.get("title") else "Unknown",
        "author": meta.get("author", "Unknown") if meta.get("author") else "Unknown",
        "pages": len(doc)
    }

def extract_images_from_pdf(file_bytes_string: bytes) -> list:
    doc = fitz.open(stream=file_bytes_string, filetype="pdf")
    extracted_images = []
    
    for page in doc:
        # Scans the mathematical layout for embedded image coordinates
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            # Convert the geometry block back to raw image bytes
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Load into Pillow Image object for Streamlit UI
            pil_image = Image.open(io.BytesIO(image_bytes))
            extracted_images.append(pil_image)
            
    return extracted_images

def extract_pages_from_pdf(file_bytes_string: bytes) -> list:
    doc = fitz.open(stream=file_bytes_string, filetype="pdf")
    pages_text = []
    
    for page in doc:
        blocks = page.get_text("blocks")
        page_width = page.rect.width
        text = sort_blocks_geometrically(blocks, page_width)
        pages_text.append(text)
        
    return pages_text

def render_pdf_page_to_image(file_bytes_string: bytes, page_number: int, highlight_texts: list = None) -> Image.Image:
    doc = fitz.open(stream=file_bytes_string, filetype="pdf")
    # Convert page_number to 0-indexed and clamp it to valid page bounds
    page_idx = max(0, min(page_number - 1, len(doc) - 1))
    page = doc[page_idx]
    
    # Apply highlights if text queries are provided
    if highlight_texts:
        import re
        for phrase in highlight_texts:
            phrase_clean = phrase.strip()
            if not phrase_clean:
                continue
            # Split paragraph chunks into sentences to ensure robust highlighting across line wraps
            sentences = re.split(r'(?<=[.!?]) +', phrase_clean)
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if len(sentence_clean) < 10:  # Skip trivial fragments
                    continue
                rects = page.search_for(sentence_clean)
                for rect in rects:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=[1.0, 0.9, 0.2])  # Warm yellow
                    annot.update()
    
    # Render page to a high-quality pixmap (dpi=150 is clear enough for reading)
    pix = page.get_pixmap(dpi=150)
    image_bytes = pix.tobytes("png")
    
    return Image.open(io.BytesIO(image_bytes))