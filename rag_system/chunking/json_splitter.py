import pymupdf.layout
import pymupdf4llm
import pymupdf
from langchain_text_splitters import TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from typing import List, Tuple, Dict
import json 
import os
from rag_system.chunking.section_classifier import SectionClassifier
from main.generator import OllamaGenerator

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

    def __init__(self, semantical_model, chunk_size: int = 600, chunk_overlap: int = 80):
        self.semantical_model = semantical_model
        self.splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, encoding_name="cl100k_base")

    def run(self, pdf_path: str):

        pages = extract_text_from_json(pdf_path)

        all_text = []
        page_mapping = []


        metadata_pdf = extract_pdf_metadata(pdf_path)
        for page in pages:
            all_text.append(page['text'])
            page_mapping.append({'page_number': page['page_number']})

        full_text = "\n\n".join(all_text)

        docs = self.splitter.create_documents([full_text])
        chunks = [d.page_content for d in docs]

        embeddings = self.semantical_model.encode(chunks, convert_to_tensor=True)

        chunk_metadata = []
        num_pages = len(page_mapping)
        chunks_per_page = max(1, len(chunks) // num_pages)

        for i, ch in enumerate(chunks):
            page_idx = min(i // chunks_per_page, num_pages - 1)
            meta = {
                'page_number': page_mapping[page_idx]['page_number']
            }
            meta.update(metadata_pdf)
            chunk_metadata.append(meta)

        return chunks, embeddings, chunk_metadata
    
    
class SemanticPDFChunker:

    def __init__(self, semantical_model=None):
        self.semantical_model = semantical_model
        
        if not self.semantical_model is None:
            self.splitter = SemanticChunker(
                SentenceTransformerEmbeddingsAdapter(semantical_model)
            )
            self.section_classifier = SectionClassifier(semantical_model)

    def _classify_chunk_with_llm(self, chunk_text: str, llm:OllamaGenerator = None) -> str:
        if not llm:
            return None
        prompt = (
            "Classifique a seção acadêmica deste trecho em uma das opções: "
            "introduction, background, method, results, discussion, conclusion"\
            "\nResponda apenas com uma única palavra da lista.\n\nTrecho:\n" + chunk_text
        )
        try:
            resp = str(llm.generate(prompt)).strip().lower()
            resp = resp.replace(".", "").strip()
            print("Classified section:", resp)
            return resp
        except Exception:
            return None

    def run(self, pdf_path: str, llm:OllamaGenerator=OllamaGenerator()) -> Tuple[List[str], any, List[Dict]]:
        pages = extract_text_from_json(pdf_path)
        metadata_pdf = extract_pdf_metadata(pdf_path)

        print("-----------------------------------------------------------------------------", metadata_pdf)
        chunks = []
        embeddings = []
        metadata = []

        for page in pages:
            text = clean_text(page["text"])

            if self._is_structural_noise(text):
                continue

            page_chunks = self._split_with_overlap(text)

            for ch in page_chunks:
                section = self._classify_chunk_with_llm(ch, llm=llm)
                chunk_meta = {
                    "page_number": page["page_number"],
                    "section": section,
                    "pdf_name": os.path.basename(pdf_path)
                }
                chunk_meta.update(metadata_pdf)
                chunks.append(ch)
                metadata.append(chunk_meta)
                print(f"Chunk added: Page {page['page_number']} | Section: {section}, Pdf: {os.path.basename(pdf_path)}")

        embeddings = self.semantical_model.encode(
            chunks,
            convert_to_tensor=True
        )

        return chunks, embeddings, metadata
    
    def _is_structural_noise(self, text: str) -> bool:
        t = text.lower()

        blacklist = [
            "sumário",
            "lista de figuras",
            "lista de tabelas",
            "referências",
            "referencias",
            "abstract",
            "resumo"
        ]

        if any(b in t for b in blacklist):
            return True

        if len(t) < 200:
            return True

        return False

    def _split_with_overlap(self, text, size=500, overlap=100):
        words = text.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk = words[i:i+size]
            chunks.append(" ".join(chunk))
            i += size - overlap

        return chunks
class SentenceTransformerEmbeddingsAdapter:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
def extract_pdf_metadata(pdf_path):
    doc = pymupdf.open(pdf_path)
    meta = doc.metadata
    return {
        "title": meta.get("title"),
        "author": meta.get("author"),
        "creationDate": meta.get("creationDate"),
        "modDate": meta.get("modDate"),
        "page_count": doc.page_count,
        "pdf_name": os.path.basename(pdf_path)
    }
