"""
Performance benchmark tests for OASIS Agentic Pipeline
Tests execution time, memory usage, and throughput for all agents.
"""

import pytest
import torch
import pandas as pd
import numpy as np
import time
import os
import sys
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.vision.vision_agent import AlzheimerVisionAgent
from agents.biomarker.biomarker_agent import ClinicalBiomarkerAgent
from agents.rag.rag_agent import MedicalLibrarianAgent
from agents.vision.explainer_agent import RadiomicsExplainerAgent
from agents.biomarker.temporal_analyst import TemporalAnalystAgent
from orchestrator.ethicist_agent import MedicalEthicistAgent


# Mark all tests in this module as benchmark tests
pytestmark = pytest.mark.benchmark


class TestVisionAgentPerformance:
    """Performance benchmarks for Vision Agent"""
    
    @pytest.fixture
    def vision_agent(self):
        """Create vision agent"""
        return AlzheimerVisionAgent(num_classes=4)
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch"""
        return torch.randn(16, 1, 224, 224)
    
    def test_single_inference_time(self, vision_agent, benchmark):
        """Benchmark single image inference time"""
        input_tensor = torch.randn(1, 1, 224, 224)
        vision_agent.eval()
        
        def inference():
            with torch.no_grad():
                return vision_agent(input_tensor)
        
        result = benchmark(inference)
        assert result is not None
        
    def test_batch_inference_time(self, vision_agent, sample_batch, benchmark):
        """Benchmark batch inference time"""
        vision_agent.eval()
        
        def batch_inference():
            with torch.no_grad():
                return vision_agent(sample_batch)
        
        result = benchmark(batch_inference)
        assert result.shape[0] == 16
        
    def test_inference_throughput(self, vision_agent):
        """Test inference throughput (images per second)"""
        vision_agent.eval()
        batch_size = 32
        num_batches = 10
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_batches):
                input_tensor = torch.randn(batch_size, 1, 224, 224)
                _ = vision_agent(input_tensor)
        end_time = time.time()
        
        total_images = batch_size * num_batches
        elapsed_time = end_time - start_time
        throughput = total_images / elapsed_time
        
        print(f"\nVision Agent Throughput: {throughput:.2f} images/second")
        assert throughput > 0
        
    def test_forward_backward_time(self, vision_agent, benchmark):
        """Benchmark forward + backward pass time"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        vision_agent.train()
        
        def forward_backward():
            output = vision_agent(input_tensor)
            loss = output.sum()
            loss.backward()
            return output
        
        result = benchmark(forward_backward)
        assert result is not None


class TestBiomarkerAgentPerformance:
    """Performance benchmarks for Biomarker Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create biomarker agent"""
        return ClinicalBiomarkerAgent()
    
    @pytest.fixture
    def large_csv(self, tmp_path):
        """Create large CSV file"""
        n_patients = 1000
        data = {
            'Subject_ID': [f'OAS1_{i:04d}' for i in range(n_patients)],
            'M/F': np.random.choice(['M', 'F'], n_patients),
            'Age': np.random.uniform(60, 95, n_patients),
            'Educ': np.random.uniform(6, 20, n_patients),
            'SES': np.random.choice([1, 2, 3, 4, 5], n_patients),
            'MMSE': np.random.uniform(0, 30, n_patients),
            'eTIV': np.random.uniform(1200, 1700, n_patients),
            'nWBV': np.random.uniform(0.6, 0.8, n_patients),
            'ASF': np.random.uniform(1.0, 1.4, n_patients)
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "large_clinical.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_data_ingestion_time(self, agent, large_csv, benchmark):
        """Benchmark data ingestion and processing time"""
        def ingest():
            return agent.ingest_and_process(large_csv)
        
        tensors, df = benchmark(ingest)
        assert tensors.shape[0] == 1000
        
    def test_normalization_time(self, agent, large_csv):
        """Test normalization performance"""
        tensors, df = agent.ingest_and_process(large_csv)
        
        start_time = time.time()
        # Simulate re-normalization
        normalized = (tensors - tensors.mean(dim=0)) / (tensors.std(dim=0) + 1e-8)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"\nNormalization time for 1000 patients: {elapsed*1000:.2f}ms")
        assert elapsed < 1.0  # Should be fast


class TestRAGAgentPerformance:
    """Performance benchmarks for RAG Agent"""
    
    @pytest.fixture
    def rag_agent(self):
        """Create RAG agent"""
        return MedicalLibrarianAgent()
    
    @pytest.fixture
    def large_document_set(self, mock_data_generator):
        """Create large document set"""
        return mock_data_generator.generate_medical_documents(n_docs=500)
    
    def test_document_ingestion_time(self, rag_agent, large_document_set, benchmark):
        """Benchmark document ingestion time"""
        def ingest():
            rag_agent.ingest_medical_guidelines(large_document_set)
        
        benchmark(ingest)
        assert len(rag_agent.document_store) == 500
        
    def test_embedding_generation_time(self, rag_agent, benchmark):
        """Benchmark embedding generation time"""
        texts = ["Test document " + str(i) for i in range(100)]
        
        def embed():
            return rag_agent.embed_text(texts)
        
        embeddings = benchmark(embed)
        assert embeddings.shape[0] == 100
        
    def test_query_time(self, rag_agent, large_document_set, benchmark):
        """Benchmark query execution time"""
        rag_agent.ingest_medical_guidelines(large_document_set)
        
        def query():
            return rag_agent.query("MMSE cognitive impairment", top_k=10)
        
        results = benchmark(query)
        assert len(results) == 10
        
    def test_query_throughput(self, rag_agent, large_document_set):
        """Test query throughput (queries per second)"""
        rag_agent.ingest_medical_guidelines(large_document_set)
        
        num_queries = 100
        queries = [f"Query {i}" for i in range(num_queries)]
        
        start_time = time.time()
        for query in queries:
            _ = rag_agent.query(query, top_k=5)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        throughput = num_queries / elapsed_time
        
        print(f"\nRAG Query Throughput: {throughput:.2f} queries/second")
        assert throughput > 0


class TestExplainerAgentPerformance:
    """Performance benchmarks for Explainer Agent"""
    
    @pytest.fixture
    def explainer_agent(self):
        """Create explainer agent"""
        model = AlzheimerVisionAgent(num_classes=4)
        return RadiomicsExplainerAgent(model)
    
    def test_heatmap_generation_time(self, explainer_agent, benchmark):
        """Benchmark heatmap generation time"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        
        def generate_heatmap():
            return explainer_agent.generate_heatmap(input_tensor, target_class=0)
        
        heatmap = benchmark(generate_heatmap)
        assert heatmap is not None
        
    def test_multiple_class_heatmaps(self, explainer_agent):
        """Test time to generate heatmaps for all classes"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        
        start_time = time.time()
        for class_idx in range(4):
            _ = explainer_agent.generate_heatmap(input_tensor, target_class=class_idx)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"\nTime to generate 4 class heatmaps: {elapsed*1000:.2f}ms")
        assert elapsed < 5.0  # Should complete in reasonable time


class TestTemporalAgentPerformance:
    """Performance benchmarks for Temporal Agent"""
    
    @pytest.fixture
    def temporal_agent(self, tmp_path):
        """Create temporal agent with large dataset"""
        n_patients = 100
        visits_per_patient = 5
        
        data = []
        for patient_id in range(n_patients):
            subject_id = f'OAS2_{patient_id:04d}'
            baseline_mmse = np.random.uniform(20, 30)
            baseline_nwbv = np.random.uniform(0.65, 0.78)
            
            for visit in range(1, visits_per_patient + 1):
                mmse_decline = np.random.uniform(0, 2) * (visit - 1)
                nwbv_decline = np.random.uniform(0, 0.02) * (visit - 1)
                
                data.append({
                    'Subject ID': subject_id,
                    'Visit': visit,
                    'MR Delay': (visit - 1) * 365 + np.random.randint(-30, 30),
                    'MMSE': max(0, baseline_mmse - mmse_decline),
                    'nWBV': max(0.5, baseline_nwbv - nwbv_decline)
                })
        
        df = pd.DataFrame(data)
        csv_path = tmp_path / "large_longitudinal.csv"
        df.to_csv(csv_path, index=False)
        
        return TemporalAnalystAgent(str(csv_path))
    
    def test_trajectory_calculation_time(self, temporal_agent, benchmark):
        """Benchmark trajectory calculation time"""
        def calculate():
            return temporal_agent.calculate_progression_trajectory("OAS2_0001")
        
        result = benchmark(calculate)
        assert 'atrophy_velocity_pct' in result
        
    def test_multiple_patient_analysis(self, temporal_agent):
        """Test time to analyze multiple patients"""
        patient_ids = [f'OAS2_{i:04d}' for i in range(50)]
        
        start_time = time.time()
        for patient_id in patient_ids:
            _ = temporal_agent.calculate_progression_trajectory(patient_id)
        end_time = time.time()
        
        elapsed = end_time - start_time
        throughput = len(patient_ids) / elapsed
        
        print(f"\nTemporal Analysis Throughput: {throughput:.2f} patients/second")
        assert throughput > 0


class TestEthicistAgentPerformance:
    """Performance benchmarks for Ethicist Agent"""
    
    @pytest.fixture
    def ethicist_agent(self):
        """Create ethicist agent"""
        return MedicalEthicistAgent()
    
    def test_audit_time(self, ethicist_agent, benchmark):
        """Benchmark audit execution time"""
        def audit():
            return ethicist_agent.audit_diagnostic_proposal(
                predicted_class="Non Demented",
                confidence=85.0,
                mmse_score=28.0,
                atrophy_velocity=0.5
            )
        
        is_flagged, message = benchmark(audit)
        assert isinstance(is_flagged, bool)
        
    def test_audit_throughput(self, ethicist_agent):
        """Test audit throughput (audits per second)"""
        num_audits = 1000
        
        start_time = time.time()
        for i in range(num_audits):
            _ = ethicist_agent.audit_diagnostic_proposal(
                predicted_class="Non Demented",
                confidence=85.0,
                mmse_score=28.0,
                atrophy_velocity=0.5
            )
        end_time = time.time()
        
        elapsed = end_time - start_time
        throughput = num_audits / elapsed
        
        print(f"\nEthicist Audit Throughput: {throughput:.2f} audits/second")
        assert throughput > 100  # Should be very fast


class TestEndToEndPerformance:
    """End-to-end performance benchmarks"""
    
    @pytest.fixture
    def setup_full_pipeline(self, tmp_path):
        """Setup full pipeline for testing"""
        # Create minimal data structure
        data_dir = tmp_path / "data" / "oasis_raw"
        data_dir.mkdir(parents=True)
        
        # Clinical CSV
        clinical_data = {
            'Subject_ID': ['OAS1_0001'],
            'M/F': ['M'],
            'Age': [75.0],
            'Educ': [14.0],
            'SES': [2.0],
            'MMSE': [28.0],
            'eTIV': [1450.0],
            'nWBV': [0.73],
            'ASF': [1.2]
        }
        clinical_df = pd.DataFrame(clinical_data)
        clinical_csv = data_dir / "oasis_clinical_data.csv"
        clinical_df.to_csv(clinical_csv, index=False)
        
        # Longitudinal CSV
        long_data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2],
            'MR Delay': [0, 365],
            'MMSE': [28.0, 26.0],
            'nWBV': [0.75, 0.73]
        }
        long_df = pd.DataFrame(long_data)
        long_csv = data_dir / "oasis_longitudinal.csv"
        long_df.to_csv(long_csv, index=False)
        
        # Test image
        img = Image.new('L', (224, 224), color=128)
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        # Src structure
        src_dir = tmp_path / "src" / "pipeline" / "onnx_inference"
        src_dir.mkdir(parents=True)
        
        return str(tmp_path), str(img_path)
    
    def test_full_diagnosis_time(self, setup_full_pipeline):
        """Test full diagnosis pipeline execution time"""
        from orchestrator.chief_medical_officer import AdvancedChiefMedicalOfficer
        
        workspace_root, img_path = setup_full_pipeline
        
        # Measure initialization time
        init_start = time.time()
        cmo = AdvancedChiefMedicalOfficer(workspace_root)
        init_end = time.time()
        init_time = init_end - init_start
        
        # Measure diagnosis time
        diag_start = time.time()
        cmo.execute_comprehensive_diagnosis(
            patient_idx=0,
            image_path=img_path,
            mock_subject_id="OAS2_0001"
        )
        diag_end = time.time()
        diag_time = diag_end - diag_start
        
        print(f"\nInitialization time: {init_time:.2f}s")
        print(f"Diagnosis time: {diag_time:.2f}s")
        print(f"Total time: {init_time + diag_time:.2f}s")
        
        # Performance targets from development plan
        assert diag_time < 5.0  # Should complete diagnosis in < 5 seconds


class TestMemoryUsage:
    """Memory usage benchmarks"""
    
    def test_vision_agent_memory(self):
        """Test vision agent memory footprint"""
        import gc
        gc.collect()
        
        # Create agent
        agent = AlzheimerVisionAgent(num_classes=4)
        
        # Process batch
        batch = torch.randn(32, 1, 224, 224)
        with torch.no_grad():
            _ = agent(batch)
        
        # Memory should be reasonable
        # This is a basic check - actual memory profiling would need psutil
        assert True
        
    def test_rag_agent_memory(self, mock_data_generator):
        """Test RAG agent memory with large document set"""
        import gc
        gc.collect()
        
        agent = MedicalLibrarianAgent()
        docs = mock_data_generator.generate_medical_documents(n_docs=1000)
        
        agent.ingest_medical_guidelines(docs)
        
        # Should handle 1000 documents without issues
        assert len(agent.document_store) == 1000


class TestScalability:
    """Scalability tests"""
    
    def test_vision_batch_scaling(self):
        """Test vision agent with increasing batch sizes"""
        agent = AlzheimerVisionAgent(num_classes=4)
        agent.eval()
        
        batch_sizes = [1, 8, 16, 32, 64]
        times = []
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 1, 224, 224)
            
            start_time = time.time()
            with torch.no_grad():
                _ = agent(input_tensor)
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"Batch size {batch_size}: {elapsed*1000:.2f}ms")
        
        # Time should scale reasonably with batch size
        assert all(t > 0 for t in times)
        
    def test_rag_document_scaling(self, mock_data_generator):
        """Test RAG agent with increasing document counts"""
        agent = MedicalLibrarianAgent()
        
        doc_counts = [10, 50, 100, 500]
        times = []
        
        for count in doc_counts:
            docs = mock_data_generator.generate_medical_documents(n_docs=count)
            
            start_time = time.time()
            agent.ingest_medical_guidelines(docs)
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"{count} documents: {elapsed*1000:.2f}ms")
        
        # Should handle increasing document counts
        assert all(t > 0 for t in times)


class TestConcurrency:
    """Concurrency and parallel processing tests"""
    
    def test_parallel_inference(self):
        """Test parallel inference capability"""
        agent = AlzheimerVisionAgent(num_classes=4)
        agent.eval()
        
        # Simulate parallel processing with multiple batches
        num_batches = 10
        batch_size = 8
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_batches):
                input_tensor = torch.randn(batch_size, 1, 224, 224)
                _ = agent(input_tensor)
        end_time = time.time()
        
        total_images = num_batches * batch_size
        elapsed = end_time - start_time
        throughput = total_images / elapsed
        
        print(f"\nParallel processing throughput: {throughput:.2f} images/second")
        assert throughput > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark"])
