"""
ONNX Export Functionality for Vision Agent
Exports trained PyTorch models to ONNX format for deployment and inference.
"""

import os
import sys
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Tuple, Optional
import argparse

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(src_dir)

from agents.vision.vision_agent import AlzheimerVisionAgent


class ONNXExporter:
    """Exports PyTorch models to ONNX format"""

    def __init__(
        self,
        model: torch.nn.Module,
        output_dir: str = "models/onnx",
        model_name: str = "vision_agent",
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Set model to evaluation mode
        self.model.eval()

    def export(
        self,
        input_shape: Tuple[int, ...] = (1, 1, 224, 224),
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None,
        optimize: bool = True,
        quantize: bool = False,
    ) -> str:
        """Export model to ONNX format"""
        print("\n" + "=" * 70)
        print("Exporting Model to ONNX")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Input shape: {input_shape}")
        print(f"Opset version: {opset_version}")
        print(f"Optimization: {optimize}")
        print(f"Quantization: {quantize}")
        print("=" * 70 + "\n")

        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Define output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        onnx_filename = f"{self.model_name}_{timestamp}.onnx"
        onnx_path = self.output_dir / onnx_filename

        # Define dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        # Export to ONNX
        print("Exporting to ONNX format...")
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )
        print(f"✓ Exported to: {onnx_path}")

        # Verify ONNX model
        print("\nVerifying ONNX model...")
        self._verify_onnx(str(onnx_path))

        # Optimize if requested
        if optimize:
            print("\nOptimizing ONNX model...")
            optimized_path = self._optimize_onnx(str(onnx_path))
            onnx_path = Path(optimized_path)

        # Quantize if requested
        if quantize:
            print("\nQuantizing ONNX model...")
            quantized_path = self._quantize_onnx(str(onnx_path))
            onnx_path = Path(quantized_path)

        # Test inference
        print("\nTesting ONNX inference...")
        self._test_inference(str(onnx_path), dummy_input)

        # Save metadata
        self._save_metadata(str(onnx_path), input_shape, opset_version)

        print("\n" + "=" * 70)
        print("Export Complete!")
        print(f"ONNX model saved to: {onnx_path}")
        print("=" * 70 + "\n")

        return str(onnx_path)

    def _verify_onnx(self, onnx_path: str):
        """Verify ONNX model"""
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model is valid")
        except Exception as e:
            print(f"✗ ONNX model verification failed: {e}")
            raise

    def _optimize_onnx(self, onnx_path: str) -> str:
        """Optimize ONNX model"""
        try:
            from onnxruntime.transformers import optimizer

            # Create optimized path
            optimized_path = onnx_path.replace(".onnx", "_optimized.onnx")

            # Optimize
            optimized_model = optimizer.optimize_model(
                onnx_path,
                model_type="bert",  # Generic optimization
                num_heads=0,
                hidden_size=0,
            )

            optimized_model.save_model_to_file(optimized_path)
            print(f"✓ Optimized model saved to: {optimized_path}")

            return optimized_path

        except Exception as e:
            print(f"⚠ Optimization failed: {e}")
            print("Continuing with unoptimized model...")
            return onnx_path

    def _quantize_onnx(self, onnx_path: str) -> str:
        """Quantize ONNX model to INT8"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            # Create quantized path
            quantized_path = onnx_path.replace(".onnx", "_int8.onnx")

            # Quantize
            quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QInt8)
            print(f"✓ Quantized model saved to: {quantized_path}")

            # Compare file sizes
            original_size = os.path.getsize(onnx_path) / (1024 * 1024)
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
            reduction = ((original_size - quantized_size) / original_size) * 100

            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Quantized size: {quantized_size:.2f} MB")
            print(f"  Size reduction: {reduction:.1f}%")

            return quantized_path

        except Exception as e:
            print(f"⚠ Quantization failed: {e}")
            print("Continuing with unquantized model...")
            return onnx_path

    def _test_inference(self, onnx_path: str, dummy_input: torch.Tensor):
        """Test ONNX inference"""
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(onnx_path)

            # Get input/output names
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            # Run inference
            onnx_input = {input_name: dummy_input.numpy()}
            onnx_output = session.run([output_name], onnx_input)[0]

            # Compare with PyTorch output
            with torch.no_grad():
                pytorch_output = self.model(dummy_input).numpy()

            # Calculate difference
            max_diff = np.abs(onnx_output - pytorch_output).max()
            mean_diff = np.abs(onnx_output - pytorch_output).mean()

            print("✓ ONNX inference successful")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")

            if max_diff > 1e-3:
                print("⚠ Warning: Large difference between PyTorch and ONNX outputs")

        except Exception as e:
            print(f"✗ ONNX inference test failed: {e}")
            raise

    def _save_metadata(self, onnx_path: str, input_shape: Tuple, opset_version: int):
        """Save export metadata"""
        metadata = {
            "model_name": self.model_name,
            "onnx_path": onnx_path,
            "input_shape": list(input_shape),
            "opset_version": opset_version,
            "exported_at": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "onnx_version": onnx.__version__,
        }

        metadata_path = onnx_path.replace(".onnx", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"✓ Metadata saved to: {metadata_path}")


class ONNXInferenceEngine:
    """ONNX Runtime inference engine"""

    def __init__(self, onnx_path: str, device: str = "cpu"):
        self.onnx_path = onnx_path
        self.device = device

        # Create session
        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        print(f"✓ Loaded ONNX model: {onnx_path}")
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_name}")
        print(f"  Providers: {self.session.get_providers()}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""
        onnx_input = {self.input_name: input_data}
        output = self.session.run([self.output_name], onnx_input)[0]
        return output

    def predict_batch(self, batch_data: np.ndarray) -> np.ndarray:
        """Run batch inference"""
        return self.predict(batch_data)

    def benchmark(self, input_shape: Tuple[int, ...], num_iterations: int = 100) -> Dict:
        """Benchmark inference performance"""
        import time

        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.predict(dummy_input)

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            self.predict(dummy_input)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time

        return {
            "total_time": total_time,
            "avg_inference_time": avg_time,
            "throughput": throughput,
            "num_iterations": num_iterations,
        }


def export_vision_agent(
    model_path: str, output_dir: str = "models/onnx", optimize: bool = True, quantize: bool = False
) -> str:
    """Export Vision Agent model to ONNX"""
    # Load model
    print("Loading Vision Agent model...")
    model = AlzheimerVisionAgent(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("✓ Model loaded")

    # Create exporter
    exporter = ONNXExporter(model, output_dir, "vision_agent")

    # Export
    onnx_path = exporter.export(
        input_shape=(1, 1, 224, 224), opset_version=14, optimize=optimize, quantize=quantize
    )

    return onnx_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Export Vision Agent to ONNX")

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained PyTorch model (.pth file)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/onnx", help="Directory to save ONNX model"
    )
    parser.add_argument("--optimize", action="store_true", default=True, help="Optimize ONNX model")
    parser.add_argument(
        "--quantize", action="store_true", default=False, help="Quantize model to INT8"
    )
    parser.add_argument(
        "--test-inference", action="store_true", help="Test ONNX inference after export"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark ONNX inference performance"
    )

    args = parser.parse_args()

    # Export model
    onnx_path = export_vision_agent(
        model_path=args.model_path,
        output_dir=args.output_dir,
        optimize=args.optimize,
        quantize=args.quantize,
    )

    # Test inference if requested
    if args.test_inference:
        print("\n" + "=" * 70)
        print("Testing ONNX Inference")
        print("=" * 70)

        engine = ONNXInferenceEngine(onnx_path)

        # Create test input
        test_input = np.random.randn(1, 1, 224, 224).astype(np.float32)

        # Run inference
        output = engine.predict(test_input)
        print(f"\nOutput shape: {output.shape}")
        print(f"Predicted class: {np.argmax(output)}")
        print(f"Confidence: {np.max(output):.4f}")

    # Benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 70)
        print("Benchmarking ONNX Inference")
        print("=" * 70)

        engine = ONNXInferenceEngine(onnx_path)

        # Benchmark
        results = engine.benchmark(input_shape=(1, 1, 224, 224), num_iterations=100)

        print("\nBenchmark Results:")
        print(f"  Total time: {results['total_time']:.4f} seconds")
        print(f"  Average inference time: {results['avg_inference_time'] * 1000:.2f} ms")
        print(f"  Throughput: {results['throughput']:.2f} inferences/second")


if __name__ == "__main__":
    main()
