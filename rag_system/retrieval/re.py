from typing import List, Dict, Optional
import re
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None
from sentence_transformers import CrossEncoder
from rag_system.chunking.section_classifier import SectionClassifier
from main.generator import OllamaGenerator
class RetrievalSystem:
    def __init__(self, vector_db, max_tokens: int = 10000):

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.vector_db = vector_db
        self.max_tokens = max_tokens
        self.section_classifier = None

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
    

    def _metadata_similarity(self, query: str, doc: Dict) -> float:
        q_lower = query.lower().strip()
        fields = []
        if doc.get("title"):
            fields.append(str(doc["title"]).lower())
        if doc.get("author"):
            fields.append(str(doc["author"]).lower())
        if not fields or not q_lower:
            return 0.0
        best = 0.0
        for f in fields:
            if not f:
                continue
            if fuzz:
                score = (fuzz.partial_ratio(q_lower, f) or 0) / 100.0
            else:
                q_tokens = set(q_lower.split())
                f_tokens = set(f.split())
                if not f_tokens or not q_tokens:
                    continue
                overlap = q_tokens.intersection(f_tokens)
                denom = max(len(q_tokens), 1)
                score = len(overlap) / denom
            best = max(best, score)
        return best

    def _rerank(self, query: str, documents: List[Dict], top_k: int = 5):
        pairs = [(query, doc["text"]) for doc in documents]

        scores = self.reranker.predict(pairs)

        blended = []
        for doc, base in zip(documents, scores):
            meta_boost = self._metadata_similarity(query, doc)
            final_score = float(base) + 0.2 * meta_boost
            blended.append((doc, final_score))

        blended.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in blended[:top_k]]

    def _keyword_search(self, query: str, all_documents: List[Dict]):
        query_terms = query.lower().split()

        hits = []
        for doc in all_documents:
            text_lower = doc["text"].lower()
            score = sum(1 for term in query_terms if term in text_lower)

            if score > 0:
                hits.append((score, doc))

        hits.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in hits]

    def _rewrite_query(self, query: str, llm: OllamaGenerator= OllamaGenerator()):
        return query
        if not llm:
            return query
        prompt = (f"""
            Você é um módulo de reescrita de consultas para busca semântica em PDFs acadêmicos.

            Sua tarefa é:
            - Reescrever a pergunta mantendo EXATAMENTE o mesmo significado
            - Tornar a pergunta mais objetiva, técnica e adequada para recuperação de trechos acadêmicos
            - NÃO explicar nada
            - NÃO listar opções
            - NÃO adicionar comentários
            - NÃO repetir a pergunta original
            - sem traduzir termos de outros idiomas para o português

            Retorne SOMENTE a pergunta reescrita em uma ÚNICA linha.

            Pergunta original:
            {query}

            Pergunta reescrita:
            """
        )
        return llm.generate(prompt)

    def _infer_section_from_query(self, query: str, semantical_model=None, llm:OllamaGenerator = OllamaGenerator()) -> Optional[str]:
        ql = query.lower()
        mapping = {
            "introduction": ["introducao", "introdução", "introduce", "intro"],
            "background": ["background", "fundamentação", "fundamentacao", "referencial", "related work", "trabalhos relacionados"],
            "method": ["metodo", "metodologia", "método", "methods", "method"],
            "results": ["resultado", "resultados", "results"],
            "discussion": ["discussao", "discussão", "discussion"],
            "conclusion": ["conclusao", "conclusão", "conclusion"],
        }

        if llm:
            prompt = (
                "Identifique a seção acadêmica alvo desta pergunta (introduction, background, method, results, discussion, conclusion).\n"
                "Responda apenas com o nome da seção.\nPergunta: " + query
            )
            try:
                resp = str(llm.generate(prompt)).lower()
                for section in mapping.keys():
                    if section in resp:
                        return section
            except Exception:
                pass
            
        for section, keywords in mapping.items():
            if any(k in ql for k in keywords):
                return section


        if semantical_model and self.section_classifier is None:
            try:
                self.section_classifier = SectionClassifier(semantical_model)
            except Exception:
                self.section_classifier = None
        if self.section_classifier:
            try:
                return self.section_classifier.classify(query)
            except Exception:
                return None
        return None
    
    def retrieve(
        self,
        query: str,
        n_results: int = 30,
        final_k: int = 5,
        semantical_model=None,
        tokenizer=None,
        llm:OllamaGenerator=OllamaGenerator(),
        all_documents_for_keyword=None,
        pdf_name: str = None,    
        section: str = None,
        author: Optional[str] = None,
        title: Optional[str] = None
    ) -> Dict[str, List[Dict]]:

        query_for_search = self._rewrite_query(query, llm=llm)
        
        print("Query for search:", query_for_search)

        query_embedding = self.vector_db.embed_query(
            query_for_search,
            semantical_model=semantical_model
        )

        print(n_results, "results will be retrieved from vector DB.")

        where_filter = {}

        if pdf_name:
            where_filter["pdf_name"] = pdf_name

        if section:
            where_filter["section"] = section

        if author:
            where_filter["author"] = author
        if title:
            where_filter["title"] = title

        if "section" not in where_filter:
            inferred_section = self._infer_section_from_query(query, semantical_model=semantical_model, llm=llm)
            if inferred_section:
                where_filter["section"] = inferred_section

        print("Using filter:", where_filter)
        dados = self.vector_db.collection.get()
            
        results = self.vector_db.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        print()

        documents = []
        docs_list = results.get('documents', [[]])[0]
        metas_list = results.get('metadatas', [[]])[0]

        for text, meta in zip(docs_list, metas_list):

            if not text:
                continue

            text = str(text).strip()
            if not text:
                continue
            
            print("section in meta:", meta.get("section") if meta else None)
            print("text", text[:100])
            
            documents.append({
                "text": text,
                "title": meta.get("title") if meta else None,
                "author": meta.get("author") if meta else None,
                "page": (meta.get("page") if meta else None) or (meta.get("page_number") if meta else None),
                "section": meta.get("section") if meta else None,
                "pdf_name": meta.get("pdf_name") if meta else None
            })

        print(len(documents), "documents retrieved from vector DB.")

        seen_texts = set()
        filtered_docs = []
        for doc in documents:
            if doc['text'] not in seen_texts:
                filtered_docs.append(doc)
                seen_texts.add(doc['text'])

        if all_documents_for_keyword:
            keyword_hits = self._keyword_search(query, all_documents_for_keyword)
            filtered_docs.extend(keyword_hits)


        reranked_docs = self._rerank(query, filtered_docs, top_k=final_k)

        if reranked_docs:
            main_pdf_name = reranked_docs[0].get("pdf_name")
            reranked_docs = [doc for doc in reranked_docs if doc.get("pdf_name") == main_pdf_name]

        truncated_docs = self._truncate_context(reranked_docs, tokenizer=tokenizer) 

        return {
            "documents": truncated_docs,
            "query_original": query,
            "query_usada": query_for_search,
            "pdf_filtrado": main_pdf_name if reranked_docs else pdf_name,
            "section_filtrada": where_filter.get("section", section),
            "author_hint": author,
            "title_hint": title
        }

