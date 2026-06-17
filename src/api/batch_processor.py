"""
Batch Processing Interface for OASIS Agentic Pipeline
Handles multiple patient diagnoses efficiently with parallel processing.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import torch
from PIL import Image
import base64
from io import BytesIO
from tqdm import tqdm

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

from orchestrator.chief_medical_officer import AdvancedChiefMedicalOfficer


class BatchProcessor:
    """Batch processing engine for multiple patient diagnoses"""
    
    def __init__(
        self,
        workspace_root: str,
        max_workers: int = 4,
        use_gpu: bool = True
    ):
        self.workspace_root = workspace_root
        self.max_workers = max_workers
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize CMO
        self.cmo = AdvancedChiefMedicalOfficer(workspace_root=workspace_root)
        
        # Results storage
        self.results_dir = Path(workspace_root) / "data" / "batch_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("✓ Batch Processor initialized")
        print(f"  Device: {self.cmo.device}")
        print(f"  Max workers: {max_workers}")
        print(f"  Results directory: {self.results_dir}")
    
    def process_single_patient(
        self,
        patient_data: Dict[str, Any],
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a single patient diagnosis"""
        try:
            patient_id = patient_data.get('patient_id', 'UNKNOWN')
            age = float(patient_data.get('age', 0))
            mmse = float(patient_data.get('mmse', 0))
            
            # Process image if provided
            vision_result = None
            explainability_result = None
            
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path).convert('L')
                img_tensor = self.cmo.image_transform(img).unsqueeze(0).to(self.cmo.device)
                
                with torch.no_grad():
                    vision_output = self.cmo.vision_agent(img_tensor)
                    probabilities = torch.nn.functional.softmax(vision_output[0], dim=0)
                
                pred_idx = int(torch.argmax(probabilities).item())
                pred_class = self.cmo.class_names[pred_idx]
                confidence = float(probabilities[pred_idx].item() * 100)
                
                vision_result = {
                    "class": pred_class,
                    "confidence": confidence,
                    "probabilities": probabilities.cpu().numpy().tolist()
                }
                
                # Generate heatmap
                heatmap = self.cmo.explainer_agent.generate_heatmap(img_tensor, target_class=pred_idx)
                explainability_result = {
                    "heatmap_available": True,
                    "peak_activation": list(heatmap.argmax())
                }
                
            elif image_base64:
                # Decode base64 image
                image_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(image_data)).convert('L')
                img_tensor = self.cmo.image_transform(img).unsqueeze(0).to(self.cmo.device)
                
                with torch.no_grad():
                    vision_output = self.cmo.vision_agent(img_tensor)
                    probabilities = torch.nn.functional.softmax(vision_output[0], dim=0)
                
                pred_idx = int(torch.argmax(probabilities).item())
                pred_class = self.cmo.class_names[pred_idx]
                confidence = float(probabilities[pred_idx].item() * 100)
                
                vision_result = {
                    "class": pred_class,
                    "confidence": confidence,
                    "probabilities": probabilities.cpu().numpy().tolist()
                }
                
                explainability_result = {"heatmap_available": False}
            else:
                vision_result = {"class": "Not provided", "confidence": 0.0}
                explainability_result = {"heatmap_available": False}
            
            # Temporal analysis
            longitudinal_id = patient_data.get('longitudinal_id')
            if longitudinal_id:
                long_metrics = self.cmo.temporal_agent.calculate_progression_trajectory(longitudinal_id)
                temporal_result = {
                    "trend": long_metrics.get('clinical_trend', 'N/A'),
                    "atrophy_velocity": float(long_metrics.get('atrophy_velocity_pct', 0.0))
                }
            else:
                temporal_result = {"trend": "N/A", "atrophy_velocity": 0.0}
            
            # RAG query
            query = f"Clinical guidelines for MMSE {mmse:.1f} and age {age:.1f}"
            rag_results = self.cmo.rag_agent.query(query, top_k=1)
            rag_context = [result[0] for result in rag_results]
            
            # Ethics audit
            is_flagged, restriction_log = self.cmo.ethicist_agent.audit_diagnostic_proposal(
                predicted_class=vision_result["class"],
                confidence=vision_result["confidence"],
                mmse_score=mmse,
                atrophy_velocity=temporal_result["atrophy_velocity"]
            )
            
            return {
                "patient_id": patient_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "success",
                "vision_prediction": vision_result,
                "temporal_analysis": temporal_result,
                "rag_context": rag_context,
                "explainability": explainability_result,
                "ethics_audit": {
                    "approved": not is_flagged,
                    "message": restriction_log
                },
                "final_diagnosis": vision_result["class"] if not is_flagged else "BLOCKED",
                "confidence": vision_result["confidence"],
                "approved": not is_flagged
            }
            
        except Exception as e:
            return {
                "patient_id": patient_data.get('patient_id', 'UNKNOWN'),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def process_batch_sequential(
        self,
        patients: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Process batch sequentially"""
        results = []
        
        iterator = tqdm(patients, desc="Processing patients") if show_progress else patients
        
        for patient_data in iterator:
            result = self.process_single_patient(patient_data)
            results.append(result)
        
        return results
    
    def process_batch_parallel(
        self,
        patients: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Process batch in parallel using ThreadPoolExecutor"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.process_single_patient, patient_data)
                for patient_data in patients
            ]
            
            if show_progress:
                from tqdm import tqdm
                for future in tqdm(futures, desc="Processing patients"):
                    results.append(future.result())
            else:
                results = [future.result() for future in futures]
        
        return results
    
    def process_csv_file(
        self,
        csv_path: str,
        image_dir: Optional[str] = None,
        parallel: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Process patients from CSV file"""
        print(f"\n{'='*70}")
        print("Batch Processing from CSV")
        print(f"{'='*70}")
        print(f"CSV file: {csv_path}")
        print(f"Image directory: {image_dir if image_dir else 'Not provided'}")
        print(f"Processing mode: {'Parallel' if parallel else 'Sequential'}")
        print(f"{'='*70}\n")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} patients from CSV")
        
        # Prepare patient data
        patients = []
        for idx, row in df.iterrows():
            patient_data = {
                'patient_id': row.get('patient_id', row.get('Subject_ID', f'P{idx}')),
                'age': float(row.get('age', row.get('Age', 0))),
                'mmse': float(row.get('mmse', row.get('MMSE', 0))),
                'longitudinal_id': row.get('longitudinal_id', None)
            }
            
            # Add image path if directory provided
            if image_dir:
                image_filename = row.get('image_filename')
                if image_filename:
                    patient_data['image_path'] = os.path.join(image_dir, image_filename)
            
            patients.append(patient_data)
        
        # Process batch
        start_time = datetime.now()
        
        if parallel:
            results = self.process_batch_parallel(patients, show_progress=True)
        else:
            results = self.process_batch_sequential(patients, show_progress=True)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        approved = sum(1 for r in results if r.get('approved', False))
        blocked = successful - approved
        
        batch_summary = {
            'total_patients': len(results),
            'successful': successful,
            'failed': failed,
            'approved': approved,
            'blocked': blocked,
            'processing_time_seconds': processing_time,
            'avg_time_per_patient': processing_time / len(results) if results else 0,
            'throughput_patients_per_hour': (len(results) / processing_time) * 3600 if processing_time > 0 else 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Save results if requested
        if save_results:
            job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            results_file = self.results_dir / f"{job_id}.json"
            
            output_data = {
                'summary': batch_summary,
                'results': results
            }
            
            with open(results_file, 'w') as f:
                json.dump(output_data, f, indent=4)
            
            print(f"\n✓ Results saved to: {results_file}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("Batch Processing Complete")
        print(f"{'='*70}")
        print(f"Total patients: {batch_summary['total_patients']}")
        print(f"Successful: {successful} ({successful/len(results)*100:.1f}%)")
        print(f"Failed: {failed}")
        print(f"Approved: {approved}")
        print(f"Blocked: {blocked}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Avg time per patient: {batch_summary['avg_time_per_patient']:.2f} seconds")
        print(f"Throughput: {batch_summary['throughput_patients_per_hour']:.1f} patients/hour")
        print(f"{'='*70}\n")
        
        return {
            'summary': batch_summary,
            'results': results
        }
    
    def export_results_to_csv(
        self,
        results: List[Dict[str, Any]],
        output_path: str
    ):
        """Export results to CSV format"""
        # Flatten results for CSV
        flattened = []
        for result in results:
            if result['status'] == 'success':
                flat_result = {
                    'patient_id': result['patient_id'],
                    'timestamp': result['timestamp'],
                    'diagnosis': result['final_diagnosis'],
                    'confidence': result['confidence'],
                    'approved': result['approved'],
                    'vision_class': result['vision_prediction']['class'],
                    'vision_confidence': result['vision_prediction']['confidence'],
                    'temporal_trend': result['temporal_analysis']['trend'],
                    'atrophy_velocity': result['temporal_analysis']['atrophy_velocity'],
                    'ethics_approved': result['ethics_audit']['approved'],
                    'ethics_message': result['ethics_audit']['message']
                }
            else:
                flat_result = {
                    'patient_id': result['patient_id'],
                    'timestamp': result['timestamp'],
                    'status': 'error',
                    'error': result.get('error', 'Unknown error')
                }
            
            flattened.append(flat_result)
        
        # Create DataFrame and save
        df = pd.DataFrame(flattened)
        df.to_csv(output_path, index=False)
        print(f"✓ Results exported to CSV: {output_path}")


def main():
    """Main entry point for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Processing for OASIS Pipeline')
    parser.add_argument('--csv', required=True, help='Path to CSV file with patient data')
    parser.add_argument('--image-dir', help='Directory containing MRI images')
    parser.add_argument('--output', help='Output file path for results')
    parser.add_argument('--parallel', action='store_true', default=True, help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV')
    
    args = parser.parse_args()
    
    # Get workspace root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Initialize processor
    processor = BatchProcessor(
        workspace_root=workspace_root,
        max_workers=args.workers
    )
    
    # Process batch
    batch_results = processor.process_csv_file(
        csv_path=args.csv,
        image_dir=args.image_dir,
        parallel=args.parallel,
        save_results=True
    )
    
    # Export to CSV if requested
    if args.export_csv:
        output_csv = args.output or f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        processor.export_results_to_csv(batch_results['results'], output_csv)


if __name__ == '__main__':
    main()
