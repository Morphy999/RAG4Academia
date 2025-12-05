from typing import List, Dict

class RetrievalSystem:
    """
    Sistema de recuperação semântica baseado em vetor embeddings.
    """

    def __init__(self, vector_db, max_tokens: int = 10000):
        self.vector_db = vector_db
        self.max_tokens = max_tokens

    def _truncate_context(self, documents: List[Dict[str, str]], tokenizer=None) -> List[Dict[str, str]]:
        if not tokenizer:
            total_tokens = 0
            truncated_docs = []
            for doc in documents:
                doc_tokens = len(doc['text'].split())
                if total_tokens + doc_tokens > self.max_tokens:
                    break
                truncated_docs.append(doc)
                total_tokens += doc_tokens
            return truncated_docs
        else:
            total_tokens = 0
            truncated_docs = []
            for doc in documents:
                doc_tokens = len(tokenizer.encode(doc['text']))
                if total_tokens + doc_tokens > self.max_tokens:
                    break
                truncated_docs.append(doc)
                total_tokens += doc_tokens
            return truncated_docs

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        semantical_model=None,
        tokenizer=None
    ) -> Dict[str, List[Dict]]:
        """
        Recupera os documentos mais relevantes para a query.
        """
        query_embedding = self.vector_db.embed_query(query, semantical_model=semantical_model)

        results = self.vector_db.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        documents = []
        docs_list = results.get('documents', [[]])[0]
        metas_list = results.get('metadatas', [[]])[0]

        for text, meta in zip(docs_list, metas_list):
            print(text, meta)

            if text is None:
                continue

            text = str(text).strip()

            if not text:
                continue

            documents.append({
                "text": text,
                "title": meta.get("title") if meta else None,
                "page": meta.get("page") if meta else None
            })

        seen_texts = set()
        filtered_docs = []
        for doc in documents:
            if doc['text'] not in seen_texts:
                filtered_docs.append(doc)
                seen_texts.add(doc['text'])

        truncated_docs = self._truncate_context(filtered_docs, tokenizer=tokenizer)

        return {"documents": truncated_docs, "query": query}
