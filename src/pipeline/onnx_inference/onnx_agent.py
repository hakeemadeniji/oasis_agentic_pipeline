import os
import numpy as np
import onnxruntime as ort
from typing import Tuple, List

class ONNXMultimodalFusionAgent:
    """
    Agent 7: ONNX Runtime Optimization Engine.
    Executes hardware-accelerated, quantized cross-attention inference
    optimized for native ARM64 / Snapdragon architecture topologies.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] Critical Hardware Alert: Target optimized asset not found at {model_path}")
            
        print(f"[*] Booting ONNX Runtime Engine... Loading asset: {os.path.basename(model_path)}")
        
        # Configure execution engine options for low-latency edge performance
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Automatically defaults to CPU execution with fallback routing structures
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        
        # Map input structural metadata names cleanly
        self.input_name_img: str = str(self.session.get_inputs()[0].name)     # 'image_mri'
        self.input_name_tab: str = str(self.session.get_inputs()[1].name)     # 'tabular_biomarkers'
        
        self.class_names: List[str] = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]

    def execute_hardware_inference(self, mri_np: np.ndarray, biomarkers_np: np.ndarray) -> Tuple[str, float]:
        """Runs accelerated forward evaluations on compiled 8-bit integer weights."""
        # Force exact batch dimension mapping: shape [1, 1, 224, 224] and [1, 8]
        if len(mri_np.shape) == 3:
            mri_np = np.expand_dims(mri_np, axis=0)
        if len(biomarkers_np.shape) == 1:
            biomarkers_np = np.expand_dims(biomarkers_np, axis=0)
            
        # Bind input dictionaries to execution threads
        inputs = {
            self.input_name_img: mri_np.astype(np.float32),
            self.input_name_tab: biomarkers_np.astype(np.float32)
        }
        
        # Compute forward execution via highly optimized C++ backend layers
        raw_outputs = self.session.run(None, inputs)
        
        # FIXED: Cast the untyped generic output list into an explicit float32 NumPy array 
        # and squeeze it to guarantee a 1D vector. This clears Pylance errors and prevents shape mismatches.
        logits_vector: np.ndarray = np.asarray(raw_outputs[0], dtype=np.float32).squeeze()
        
        # Implement a pure NumPy Stable Softmax operation to isolate prediction confidences
        exp_logits = np.exp(logits_vector - np.max(logits_vector))
        probabilities = exp_logits / np.sum(exp_logits)
        
        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx] * 100)
        
        return self.class_names[pred_idx], confidence