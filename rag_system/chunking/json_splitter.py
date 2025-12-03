import pymupdf.layout
import pymupdf4llm
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple, Dict
import json 


def extract_text_from_json(pdf_path: str) -> List[Dict]:

    doc = pymupdf.open(pdf_path)

    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        raw_text = page.get_text().strip()
        if not raw_text:
            continue

        try:
            json_str = pymupdf4llm.to_json(
                doc,
                pages=[page_num],
                header=False,
                footer=False
            )
        except Exception as e:
            print(f"[WARN] Falha extraindo layout da página {page_num} ({pdf_path}): {e}")
            pages_text.append({
                'page_number': page_num,
                'text': raw_text
            })
            continue

        data = json.loads(json_str)

        lines_output = []
        fulltext = data["pages"][0].get("fulltext", [])

        for ft in fulltext:
            for line in ft.get("lines", []):
                span_texts = [s.get("text", "") for s in line.get("spans", [])]
                line_text = "".join(span_texts).strip()
                if line_text:
                    lines_output.append(line_text)

        final_text = "\n".join(lines_output) if lines_output else raw_text

        pages_text.append({
            'page_number': page_num,
            'text': final_text
        })

    return pages_text

def clean_text(t: str) -> str:
    t = t.replace("\r", "")
    t = t.replace("\n\n", "\n")
    t = t.replace("\n", " ")
    t = " ".join(t.split())
    return t


def merge_small_chunks(chunks: List[str], min_chars: int = 300) -> List[str]:
    """
    Junta chunks pequenos em blocos maiores para evitar fragmentos muito curtos.
    """
    merged = []
    buffer = ""

    chunks = [clean_text(ch) for ch in chunks]

    for ch in chunks:
        if len(ch) < min_chars:
            buffer += " " + ch
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(ch)

    if buffer:
        merged.append(buffer.strip())

    return merged


class RecursivePDFChunker:

    def __init__(self, semantical_model, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.semantical_model = semantical_model
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", ". ", "? ", "! ", "\n", " "]
        )

    def run(self, pdf_path: str) -> Tuple[List[str], List, List[Dict]]:

        pages = extract_text_from_json(pdf_path)

        all_text = []
        page_mapping = []

        for page in pages:
            all_text.append(page['text'])
            page_mapping.append({'page_number': page['page_number']})

        full_text = "\n\n".join(all_text)

        docs = self.splitter.create_documents([full_text])
        chunks = [d.page_content for d in docs]

        chunks = merge_small_chunks(chunks, min_chars=300)
        
        embeddings = self.semantical_model.encode(chunks, convert_to_tensor=True)

        chunk_metadata = []
        num_pages = len(page_mapping)
        chunks_per_page = max(1, len(chunks) // num_pages)

        for i, ch in enumerate(chunks):
            page_idx = min(i // chunks_per_page, num_pages - 1)
            chunk_metadata.append({'page_number': page_mapping[page_idx]['page_number']})
            
        return chunks, embeddings, chunk_metadata


# if __name__ == "__main__":
    # semantical_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

#     chunker = RecursivePDFChunker(
#         semantical_model,
#         chunk_size=1600,
#         chunk_overlap=100
#     )

#     pdf_path = r"C:\Users\dracb\OneDrive\Documentos\GitHub\RAG4Academia\data\DissertacaoRobertavFinal.pdf"

#     chunks, embeddings, metadata = chunker.run(pdf_path)
    
#     output_dir = "chunks_recursive_json"
#     os.makedirs(output_dir, exist_ok=True)

#     for i, chunk in enumerate(chunks, start=1):
#         filename = os.path.join(output_dir, f"chunk_{i}.txt")
#         with open(filename, "w", encoding="utf-8") as f:
#             f.write(chunk)
#         print(f"Chunk {i} salvo em: {filename} (página: {metadata[i-1]['page_number']})")

#     print(f"Total de chunks: {len(chunks)}")
