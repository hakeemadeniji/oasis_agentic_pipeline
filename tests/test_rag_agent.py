"""
Unit tests for RAG Agent (Agent 3) - Medical Librarian
Tests document ingestion, embedding generation, and semantic search functionality.
"""

import pytest
import torch
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.rag.rag_agent import MedicalLibrarianAgent


class TestMedicalLibrarianAgent:
    """Test suite for RAG Agent initialization and basic functionality"""

    @pytest.fixture
    def rag_agent(self):
        """Create a RAG agent instance for testing"""
        return MedicalLibrarianAgent()

    @pytest.fixture
    def sample_documents(self):
        """Create sample medical documents"""
        return [
            "MMSE scores below 24 indicate cognitive impairment.",
            "Hippocampal atrophy is a key biomarker for Alzheimer's disease.",
            "The APOE e4 allele increases risk of Alzheimer's disease.",
            "Moderate dementia is characterized by MMSE scores between 13-20.",
            "Very mild dementia typically presents with MMSE scores above 25.",
        ]

    def test_agent_initialization(self, rag_agent):
        """Test that agent initializes correctly"""
        assert rag_agent is not None
        assert rag_agent.tokenizer is not None
        assert rag_agent.model is not None
        assert rag_agent.document_store == []
        assert rag_agent.vector_store is None

    def test_model_in_eval_mode(self, rag_agent):
        """Test that model is in evaluation mode"""
        assert not rag_agent.model.training

    def test_empty_document_store(self, rag_agent):
        """Test initial document store is empty"""
        assert len(rag_agent.document_store) == 0
        assert rag_agent.vector_store is None


class TestDocumentIngestion:
    """Test suite for document ingestion functionality"""

    @pytest.fixture
    def rag_agent(self):
        return MedicalLibrarianAgent()

    @pytest.fixture
    def sample_documents(self):
        return [
            "MMSE scores below 24 indicate cognitive impairment.",
            "Hippocampal atrophy is a key biomarker for Alzheimer's disease.",
            "The APOE e4 allele increases risk of Alzheimer's disease.",
        ]

    def test_ingest_single_document(self, rag_agent):
        """Test ingesting a single document"""
        docs = ["MMSE scores below 24 indicate cognitive impairment."]
        rag_agent.ingest_medical_guidelines(docs)

        assert len(rag_agent.document_store) == 1
        assert rag_agent.vector_store is not None
        assert rag_agent.vector_store.shape[0] == 1

    def test_ingest_multiple_documents(self, rag_agent, sample_documents):
        """Test ingesting multiple documents"""
        rag_agent.ingest_medical_guidelines(sample_documents)

        assert len(rag_agent.document_store) == len(sample_documents)
        assert rag_agent.vector_store.shape[0] == len(sample_documents)

    def test_incremental_ingestion(self, rag_agent):
        """Test adding documents incrementally"""
        # First batch
        batch1 = ["Document 1", "Document 2"]
        rag_agent.ingest_medical_guidelines(batch1)
        assert len(rag_agent.document_store) == 2

        # Second batch
        batch2 = ["Document 3", "Document 4"]
        rag_agent.ingest_medical_guidelines(batch2)
        assert len(rag_agent.document_store) == 4
        assert rag_agent.vector_store.shape[0] == 4

    def test_document_store_preservation(self, rag_agent, sample_documents):
        """Test that original documents are preserved"""
        rag_agent.ingest_medical_guidelines(sample_documents)

        for i, doc in enumerate(sample_documents):
            assert rag_agent.document_store[i] == doc


class TestEmbeddingGeneration:
    """Test suite for text embedding functionality"""

    @pytest.fixture
    def rag_agent(self):
        return MedicalLibrarianAgent()

    def test_embed_single_text(self, rag_agent):
        """Test embedding a single text"""
        texts = ["MMSE scores below 24 indicate cognitive impairment."]
        embeddings = rag_agent.embed_text(texts)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0  # Embedding dimension

    def test_embed_multiple_texts(self, rag_agent):
        """Test embedding multiple texts"""
        texts = [
            "MMSE scores below 24 indicate cognitive impairment.",
            "Hippocampal atrophy is a key biomarker.",
            "The APOE e4 allele increases risk.",
        ]
        embeddings = rag_agent.embed_text(texts)

        assert embeddings.shape[0] == len(texts)

    def test_embedding_normalization(self, rag_agent):
        """Test that embeddings are L2 normalized"""
        texts = ["Test document for normalization."]
        embeddings = rag_agent.embed_text(texts)

        # Check L2 norm is approximately 1
        norm = torch.norm(embeddings[0], p=2)
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_embedding_consistency(self, rag_agent):
        """Test that same text produces same embedding"""
        text = ["MMSE scores below 24 indicate cognitive impairment."]

        embedding1 = rag_agent.embed_text(text)
        embedding2 = rag_agent.embed_text(text)

        assert torch.allclose(embedding1, embedding2, atol=1e-6)

    def test_embedding_no_nan(self, rag_agent):
        """Test embeddings don't contain NaN values"""
        texts = ["Test document."]
        embeddings = rag_agent.embed_text(texts)

        assert not torch.isnan(embeddings).any()

    def test_embedding_no_inf(self, rag_agent):
        """Test embeddings don't contain infinite values"""
        texts = ["Test document."]
        embeddings = rag_agent.embed_text(texts)

        assert not torch.isinf(embeddings).any()


class TestSemanticSearch:
    """Test suite for semantic search functionality"""

    @pytest.fixture
    def rag_agent_with_docs(self):
        """Create RAG agent with pre-loaded documents"""
        agent = MedicalLibrarianAgent()
        docs = [
            "MMSE scores below 24 indicate cognitive impairment.",
            "Hippocampal atrophy is a key biomarker for Alzheimer's disease.",
            "The APOE e4 allele increases risk of Alzheimer's disease.",
            "Moderate dementia is characterized by MMSE scores between 13-20.",
            "Very mild dementia typically presents with MMSE scores above 25.",
        ]
        agent.ingest_medical_guidelines(docs)
        return agent

    def test_query_returns_results(self, rag_agent_with_docs):
        """Test that query returns results"""
        results = rag_agent_with_docs.query("What is MMSE?", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_query_result_format(self, rag_agent_with_docs):
        """Test query result format (document, score)"""
        results = rag_agent_with_docs.query("MMSE scores", top_k=1)

        doc, score = results[0]
        assert isinstance(doc, str)
        assert isinstance(score, float)

    def test_query_score_range(self, rag_agent_with_docs):
        """Test that similarity scores are in valid range"""
        results = rag_agent_with_docs.query("MMSE", top_k=3)

        for doc, score in results:
            assert -1.0 <= score <= 1.0

    def test_query_score_ordering(self, rag_agent_with_docs):
        """Test that results are ordered by relevance"""
        results = rag_agent_with_docs.query("MMSE scores", top_k=3)

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_semantic_relevance(self, rag_agent_with_docs):
        """Test that semantically relevant documents rank higher"""
        results = rag_agent_with_docs.query("cognitive impairment MMSE", top_k=2)

        # First result should contain "MMSE" or "cognitive"
        top_doc = results[0][0].lower()
        assert "mmse" in top_doc or "cognitive" in top_doc

    def test_query_top_k_limit(self, rag_agent_with_docs):
        """Test that top_k parameter limits results"""
        results_k2 = rag_agent_with_docs.query("Alzheimer's", top_k=2)
        results_k4 = rag_agent_with_docs.query("Alzheimer's", top_k=4)

        assert len(results_k2) == 2
        assert len(results_k4) == 4

    def test_query_exceeds_document_count(self, rag_agent_with_docs):
        """Test query when top_k exceeds document count"""
        # Agent has 5 documents, request 10
        results = rag_agent_with_docs.query("test", top_k=10)

        # Should return all 5 documents
        assert len(results) <= 5

    def test_query_empty_database_raises_error(self):
        """Test that querying empty database raises error"""
        agent = MedicalLibrarianAgent()

        with pytest.raises(ValueError, match="Database is empty"):
            agent.query("test query")


class TestEdgeCases:
    """Test suite for edge cases and error handling"""

    @pytest.fixture
    def rag_agent(self):
        return MedicalLibrarianAgent()

    def test_empty_document_ingestion(self, rag_agent):
        """Test ingesting empty document list"""
        rag_agent.ingest_medical_guidelines([])

        assert len(rag_agent.document_store) == 0
        assert rag_agent.vector_store is None

    def test_very_long_document(self, rag_agent):
        """Test handling very long documents (truncation)"""
        long_doc = "word " * 1000  # Very long document
        rag_agent.ingest_medical_guidelines([long_doc])

        assert len(rag_agent.document_store) == 1
        assert rag_agent.vector_store is not None

    def test_special_characters(self, rag_agent):
        """Test handling documents with special characters"""
        docs = ["MMSE: 24/30 (80%)", "Patient #123 - Diagnosis: AD", "Dosage: 10mg/day @ 8:00 AM"]
        rag_agent.ingest_medical_guidelines(docs)

        results = rag_agent.query("MMSE score", top_k=1)
        assert len(results) == 1

    def test_unicode_characters(self, rag_agent):
        """Test handling documents with unicode characters"""
        docs = [
            "Alzheimer's disease affects memory",
            "β-amyloid plaques are a hallmark",
            "Tau protein tangles contribute to neurodegeneration",
        ]
        rag_agent.ingest_medical_guidelines(docs)

        results = rag_agent.query("amyloid", top_k=1)
        assert len(results) == 1

    def test_duplicate_documents(self, rag_agent):
        """Test handling duplicate documents"""
        docs = [
            "MMSE scores below 24 indicate impairment.",
            "MMSE scores below 24 indicate impairment.",  # Duplicate
            "Different document about Alzheimer's.",
        ]
        rag_agent.ingest_medical_guidelines(docs)

        # Should store all documents including duplicates
        assert len(rag_agent.document_store) == 3

    def test_query_with_empty_string(self, rag_agent):
        """Test querying with empty string"""
        docs = ["Test document"]
        rag_agent.ingest_medical_guidelines(docs)

        results = rag_agent.query("", top_k=1)
        assert len(results) == 1


class TestVectorStoreProperties:
    """Test suite for vector store properties"""

    @pytest.fixture
    def rag_agent(self):
        return MedicalLibrarianAgent()

    def test_vector_store_shape(self, rag_agent):
        """Test vector store has correct shape"""
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        rag_agent.ingest_medical_guidelines(docs)

        assert rag_agent.vector_store.shape[0] == 3
        assert rag_agent.vector_store.shape[1] > 0  # Embedding dimension

    def test_vector_store_dtype(self, rag_agent):
        """Test vector store has correct dtype"""
        docs = ["Test document"]
        rag_agent.ingest_medical_guidelines(docs)

        assert rag_agent.vector_store.dtype == torch.float32

    def test_vector_store_device(self, rag_agent):
        """Test vector store is on correct device"""
        docs = ["Test document"]
        rag_agent.ingest_medical_guidelines(docs)

        # Should be on CPU by default
        assert rag_agent.vector_store.device.type == "cpu"


class TestPerformance:
    """Test suite for performance characteristics"""

    @pytest.fixture
    def rag_agent(self):
        return MedicalLibrarianAgent()

    @pytest.mark.slow
    def test_large_document_set(self, rag_agent, mock_data_generator):
        """Test handling large document sets"""
        docs = mock_data_generator.generate_medical_documents(n_docs=100)
        rag_agent.ingest_medical_guidelines(docs)

        assert len(rag_agent.document_store) == 100
        assert rag_agent.vector_store.shape[0] == 100

        # Test query still works
        results = rag_agent.query("test query", top_k=5)
        assert len(results) == 5

    @pytest.mark.benchmark
    def test_query_speed(self, rag_agent, mock_data_generator):
        """Test query execution speed"""
        import time

        docs = mock_data_generator.generate_medical_documents(n_docs=50)
        rag_agent.ingest_medical_guidelines(docs)

        start_time = time.time()
        results = rag_agent.query("test query", top_k=5)
        end_time = time.time()

        query_time = end_time - start_time

        # Query should complete in reasonable time (< 1 second)
        assert query_time < 1.0
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
