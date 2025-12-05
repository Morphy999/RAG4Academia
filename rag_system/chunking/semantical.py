import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import unicodedata
import re


def dedupe_preserve_order(lines):
    out = []
    last = None
    for l in lines:
        if l.strip() != last:
            out.append(l)
        last = l.strip()
    return out

def dedupe_text_block(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(dedupe_preserve_order(lines))


class SemanticalChunker:    
    def __init__(self, semantical_model):
        self.semantical_model = semantical_model
    
    def run(
        self, 
        text: str, 
        max_chars: int = 800, 
        overlap_lines: int = 2,
        min_chars: int = 300,   
        min_lines: int = 2       
    ):
        
        raw_lines = text.split("\n")
        lines = [l.strip() for l in raw_lines if l.strip()]
        
        lines = dedupe_preserve_order(lines)

        chunks = []
        chunk = []
        current_len = 0

        for line in lines:
            line_len = len(line)

            if chunk and chunk[-1].strip() == line.strip():
                continue

            if current_len + line_len > max_chars:
                clean_chunk = dedupe_text_block("\n".join(chunk))
                chunks.append(clean_chunk)
                
                if overlap_lines > 0 and len(chunk) >= overlap_lines:
                    chunk = chunk[-overlap_lines:]
                else:
                    chunk = []

                current_len = sum(len(l) for l in chunk)

            chunk.append(line)
            current_len += line_len


        if chunk:
            clean_chunk = dedupe_text_block("\n".join(chunk))
            chunks.append(clean_chunk)

        final_chunks = []

        for c in chunks:
            c = dedupe_text_block(c)

            num_lines = len(c.split("\n"))
            num_chars = len(c)

            small = num_lines < min_lines or num_chars < min_chars

            if small:
                if final_chunks:
                    final_chunks[-1] += "\n" + c
                    final_chunks[-1] = dedupe_text_block(final_chunks[-1])
                else:
                    final_chunks.append(c)
            else:
                final_chunks.append(c)

        embeddings = self.semantical_model.encode(final_chunks, convert_to_tensor=True)
        
        return final_chunks, embeddings