import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional


class MedicalLibrarianAgent:
    """
    Pure-PyTorch RAG (Retrieval-Augmented Generation) Agent.
    Bypasses external C++ Vector DBs for flawless ARM64 compatibility.
    Fully type-calibrated for strict Pylance compliance.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[*] Booting Medical Librarian with Language Model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.document_store: List[str] = []
        # FIXED: Use Optional[torch.Tensor] to allow it to be None initially
        self.vector_store: Optional[torch.Tensor] = None

    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_text(self, texts: List[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        return F.normalize(sentence_embeddings, p=2, dim=1)

    def ingest_medical_guidelines(self, texts: List[str]):
        print(f"[*] Ingesting {len(texts)} clinical documents into Vector Database...")
        self.document_store.extend(texts)

        new_embeddings = self.embed_text(texts)

        if self.vector_store is None:
            self.vector_store = new_embeddings
        else:
            self.vector_store = torch.cat((self.vector_store, new_embeddings), dim=0)

        print(f"[+] Vector Database Updated. Shape: {self.vector_store.shape}")

    def query(self, search_text: str, top_k: int = 2) -> List[Tuple[str, float]]:
        if self.vector_store is None:
            raise ValueError("[!] Database is empty. Ingest documents first.")

        query_vector = self.embed_text([search_text])

        cosine_scores = torch.mm(query_vector, self.vector_store.transpose(0, 1))[0]
        top_results = torch.topk(cosine_scores, k=min(top_k, len(self.document_store)))

        # FIXED: Explicitly define the list type to satisfy the linter
        results: List[Tuple[str, float]] = []

        # FIXED: Cast the PyTorch items to strict Python primitives
        for score, idx in zip(top_results.values, top_results.indices):
            doc_idx = int(idx.item())
            doc_score = float(score.item())
            results.append((self.document_store[doc_idx], doc_score))

        return results


if __name__ == "__main__":
    # Synthetic Medical Knowledge Base for the Agent to memorize
    clinical_guidelines = [
        "A Mini-Mental State Examination (MMSE) score below 24 indicates potential cognitive impairment. Scores between 20-24 suggest mild dementia.",
        "Moderate dementia is often characterized by an MMSE score between 13 and 20, accompanied by severe memory loss regarding recent events.",
        "The APOE e4 allele is a major genetic risk factor for Alzheimer's disease, significantly increasing the likelihood of amyloid plaque buildup.",
        "Significant atrophy in the hippocampus and expansion of the lateral ventricles on a T1 MRI are primary structural biomarkers for Alzheimer's progression.",
        "Very Mild Dementia (CDR 0.5) usually presents as slight memory complaints, mostly forgetting names or misplacing objects, with an MMSE typically above 25.",
    ]

    librarian = MedicalLibrarianAgent()
    librarian.ingest_medical_guidelines(clinical_guidelines)

    print("\n--- Medical RAG Query Test ---")
    test_query = "What does an MMSE score of 18 mean for a patient?"
    print(f"QUERY: '{test_query}'\n")

    results = librarian.query(test_query, top_k=2)
    for rank, (doc, confidence) in enumerate(results, 1):
        print(f"Rank {rank} (Confidence: {confidence:.3f}):\n{doc}\n")
