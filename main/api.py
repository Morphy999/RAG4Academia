import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
import shutil
import os
from typing import List
from pathlib import Path

from hydra_utils.utils import instantiate_tree

api = FastAPI(title="RAG + Ollama API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class PromptRequest(BaseModel):
    prompt: str
    num_docs: int = 5

config_path = Path("hydra_utils") / "rag_system.yaml"

from omegaconf import OmegaConf

cfg_yaml = OmegaConf.load(config_path)

instances = instantiate_tree(cfg_yaml)

rag_pipeline = instances.get("rag_pipeline")

print("PIPE:", rag_pipeline, type(rag_pipeline))

UPLOAD_DIR = Path("data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@api.post("/ask_ollama3_with_rag_endpoint")
async def ask_ollama3_with_rag_endpoint(request: PromptRequest):

    final_prompt, retrievals = rag_pipeline.run(
        query_user=request.prompt,
        final_k=request.num_docs
    )
    
    print("final_prompt", final_prompt)

    generator = instances.get("generator")

    def stream():
        for token in generator.generate_stream(final_prompt):
            yield token

    return StreamingResponse(stream(), media_type="text/plain")


from fastapi.responses import StreamingResponse

@api.post("/ask_ollama3")
async def ask_ollama3_stream(request: PromptRequest):

    def generate():
        for chunk in ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": request.prompt}],
            stream=True
        ):
            yield chunk["message"]["content"]

    return StreamingResponse(generate(), media_type="text/plain")

@api.post("/upload_docs")
async def upload_docs(files: List[UploadFile] = File(...)):

    saved_files = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_files.append(file.filename)

    return {
        "status": "ok",
        "files": saved_files
    }


import hashlib

@api.post("/process_docs")
def process_docs():

    vector_db = instances.get("vector_db")        
    semantical_chunker = instances.get("semantical_chunker")

    total_chunks = 0
    processed_files = []

    for filename in os.listdir(UPLOAD_DIR):
        file_path = UPLOAD_DIR / filename

        if not filename.lower().endswith(".pdf"):
            continue

        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if filename.lower().endswith(".pdf"):
            chunks, embeddings, metadata = semantical_chunker.run(str(file_path))
            
        ids = [
            f"{filename}_{file_hash}_{i}"
            for i in range(len(chunks))
        ]

        final_metadata = []
        for meta in metadata:
            merged = dict(meta) if isinstance(meta, dict) else {}
            if "page" not in merged and "page_number" in merged:
                merged["page"] = merged.get("page_number")
            merged.update({
                "source": filename,
                "file_hash": file_hash
            })
            final_metadata.append(merged)

        vector_db.add_embeddings(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=final_metadata
        )

        total_chunks += len(chunks)
        processed_files.append(filename)

    if total_chunks == 0:
        return {
            "status": "Nenhum documento válido para processamento"
        }

    return {
        "status": "Base vetorial atualizada com sucesso",
        "processed_files": processed_files,
        "total_chunks": total_chunks
    }