import chromadb
import torch

class VectorDB:
    def __init__(self, persist_directory="chroma_db", collection_name="ufv_documents"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_embeddings(self, ids, embeddings, metadatas, documents=None):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy().tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_embeddings, n_results=5):
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy().tolist()

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        return results

    def embed_query(self, query, semantical_model=None):
        if semantical_model is None:
            raise ValueError("A semantical_model must be provided to embed the query.")

        embedding = semantical_model.encode([query], convert_to_tensor=True)

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        return embedding[0].tolist()