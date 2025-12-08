import torch
import torch.nn.functional as F
SECTION_LABELS = {
    "introduction": "Introdução do trabalho e contextualização do problema",
    "background": "Referencial teórico e trabalhos relacionados, fundamentação teorica",
    "method": "Metodologia, modelo, arquitetura e experimento",
    "results": "Resultados experimentais, métricas e avaliação",
    "discussion": "Discussão dos resultados",
    "conclusion": "Conclusão do trabalho e trabalhos futuros"
}

class SectionClassifier:
    def __init__(self, semantical_model):
        self.model = semantical_model
        self.labels = list(SECTION_LABELS.keys())
        self.label_embeddings = self.model.encode(
            list(SECTION_LABELS.values()),
            convert_to_tensor=True
        )

    def classify(self, chunk_text: str) -> str:
        chunk_emb = self.model.encode(chunk_text, convert_to_tensor=True)

        sims = F.cosine_similarity(
            chunk_emb.unsqueeze(0),
            self.label_embeddings
        )

        best_idx = torch.argmax(sims).item()
        return self.labels[best_idx]