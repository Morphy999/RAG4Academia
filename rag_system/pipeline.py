from rag_system.retrieval.re import RetrievalSystem

class RagSystemPipeline:
    
    def __init__(self, retriever: RetrievalSystem, semantical_model=None):
        self.retriever = retriever
        self.semantical_model = semantical_model
        
    def run(self, query_user: str, n_retrievals: int = 30, final_k: int = 5):
    
        retrieval_results = self.retriever.retrieve(query_user, n_results=n_retrievals,final_k=final_k, semantical_model=self.semantical_model)
        
        docs = retrieval_results.get('documents', [])
        
        if len(docs) == 0:
            context_combined = "Nenhum contexto encontrado."
        else:
            contexts = [doc['text'] for doc in retrieval_results.get('documents', [])]
            print("----------------------------------------------------------------------------")
            print(contexts)
            context_combined = "\n\n".join(contexts)

        prompt = f"""
        Use apenas UM único trecho como base para a resposta.
        Não utilize nenhum conhecimento externo ao trecho fornecido.
        Não faça inferências, deduções ou suposições.

        Se mais de um trecho mencionar modelos diferentes,
        escolha apenas aquele que estiver explicitamente descrito como utilizado no trabalho.

        Ao final da resposta, informe de onde o trecho foi retirado,
        incluindo o título do trabalho e o número da página, se disponível.

        Use o contexto a seguir para responder à pergunta.

        Contexto:
        {context_combined}
        
        Pergunta:
        {query_user}

        Resposta:
        """
        
        return prompt, retrieval_results
