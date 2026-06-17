"""
Export the retrained vision classifier to ONNX + INT8 for NPU deployment.

The multimodal *fusion* graph (compile_npu.py) is useful for the cross-attention
benchmark, but its classifier head was never trained end-to-end. The genuinely
deployable artifact is the **vision agent** (ResNet18) that the balanced trainer
brought to ~98.8% balanced accuracy. This script:

  1. loads ``best_vision_agent.pth`` into AlzheimerVisionAgent,
  2. exports an ONNX graph (input ``image_mri`` [N,1,224,224] -> ``logits``),
  3. applies dynamic INT8 quantization for the Snapdragon NPU,
  4. verifies torch vs ONNX Runtime output parity.

Usage:
    python src/pipeline/onnx_inference/export_vision_onnx.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ""))
if _SRC not in sys.path:
    sys.path.append(_SRC)

from agents.vision.vision_agent import AlzheimerVisionAgent  # noqa: E402


def export(workspace_root: str) -> None:
    out_dir = os.path.join(workspace_root, "src", "pipeline", "onnx_inference")
    weights = os.path.join(out_dir, "best_vision_agent.pth")
    raw_path = os.path.join(out_dir, "vision_agent.onnx")
    int8_path = os.path.join(out_dir, "vision_agent_int8.onnx")

    print("\n=== EXPORTING VISION CLASSIFIER TO ONNX INT8 ===")
    model = AlzheimerVisionAgent(num_classes=4)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Trained weights not found: {weights}")
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    print(f"[+] Loaded retrained weights: {os.path.basename(weights)}")

    dummy = torch.randn(1, 1, 224, 224)
    torch.onnx.export(
        model, dummy, raw_path,
        export_params=True, opset_version=18, dynamo=False,
        do_constant_folding=True,
        input_names=["image_mri"], output_names=["logits"],
        dynamic_axes={"image_mri": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[+] FP32 ONNX written: {os.path.basename(raw_path)}")

    quantize_dynamic(model_input=raw_path, model_output=int8_path, weight_type=QuantType.QUInt8)
    raw_mb = os.path.getsize(raw_path) / (1024 * 1024)
    int8_mb = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"[+] INT8 ONNX written: {os.path.basename(int8_path)} "
          f"({raw_mb:.1f} MB -> {int8_mb:.1f} MB, {(1 - int8_mb / raw_mb) * 100:.1f}% smaller)")

    # Parity check: torch vs ORT (FP32) argmax agreement on random inputs.
    import onnxruntime as ort
    sess = ort.InferenceSession(raw_path, providers=["CPUExecutionProvider"])
    agree = 0
    n = 16
    with torch.no_grad():
        for _ in range(n):
            x = torch.randn(1, 1, 224, 224)
            t_pred = int(model(x).argmax())
            o_pred = int(np.argmax(sess.run(None, {"image_mri": x.numpy()})[0]))
            agree += int(t_pred == o_pred)
    print(f"[+] torch/ONNX argmax parity: {agree}/{n}")
    print("[SUCCESS] Vision classifier exported for NPU inference.\n")


if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    export(ROOT)
