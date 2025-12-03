import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path
from rag_system.utils.preprocess import TextPreprocessor

def extract_pdf(path_pdf: Path):
    import pymupdf
    
    TextPreprocessorInstance = TextPreprocessor()
    
    doc = pymupdf.open(path_pdf.with_suffix(".pdf"))
    full_text = ""

    for page in doc:
        full_text += page.get_text() + "\n"

    full_text = TextPreprocessorInstance.run(full_text)

    with open("output.txt", "w", encoding="utf8") as out:
        out.write(full_text)

    return full_text

    
if '__main__' == __name__:
    extract_pdf(Path(r'C:\Users\dracb\OneDrive\Documentos\GitHub\RAG4Academia\data\DissertacaoRobertavFinal'))
    