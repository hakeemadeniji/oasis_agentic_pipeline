"""
Lightweight edge console server for the OASIS Clinical Console.

This is a dependency-light launcher (Python stdlib http.server only) that serves
the production web console and a real /diagnose endpoint using just the *edge*
agents -- vision (ResNet18), Grad-CAM explainer, regional volumetry, ethicist,
and the Ollama-backed reasoner (template fallback offline). It deliberately does
NOT load the RAG/transformers stack, so it boots on minimal edge devices
(e.g. a Snapdragon laptop or a kiosk wired to a hospital display / TV) where the
full FastAPI service is overkill.

For the full service (RAG + batch + Prometheus) use src/api/main.py with uvicorn.

Usage:
    python scripts/console_server.py --port 8800
    # then open http://localhost:8800/app/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
WEB = os.path.join(ROOT, "web")
if SRC not in sys.path:
    sys.path.append(SRC)

from config import get_settings  # noqa: E402
from agents.vision.vision_agent import AlzheimerVisionAgent  # noqa: E402
from agents.vision.explainer_agent import RadiomicsExplainerAgent  # noqa: E402
from agents.biomarker.volumetry_agent import RegionalVolumetryAgent  # noqa: E402
from agents.biomarker.atn_classifier import ATNBiomarkerProfiler  # noqa: E402
from agents.biomarker.pet_pup import PUPPetParser  # noqa: E402
from orchestrator.ethicist_agent import MedicalEthicistAgent  # noqa: E402
from agents.llm.llm_reasoner import ClinicalReasonerAgent  # noqa: E402
from api.heatmap import render_gradcam  # noqa: E402

CLASS_NAMES = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]
_MIME = {".html": "text/html", ".css": "text/css", ".js": "application/javascript",
         ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon"}


class Engine:
    """Lazily-initialized edge agent bundle."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = torch.device(self.settings.resolve_device())
        self.model = AlzheimerVisionAgent(num_classes=4).to(self.device)
        weights = os.path.join(ROOT, "src", "pipeline", "onnx_inference", "best_vision_agent.pth")
        if os.path.exists(weights):
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
            print(f"[+] Vision weights loaded: {os.path.basename(weights)}")
        self.model.eval()
        self.explainer = RadiomicsExplainerAgent(self.model)
        fs_root = self.settings.freesurfer_root
        if not os.path.isabs(fs_root):
            fs_root = os.path.join(ROOT, fs_root)
        self.volumetry = RegionalVolumetryAgent(freesurfer_root=fs_root)
        self.ethicist = MedicalEthicistAgent(confidence_floor=self.settings.confidence_floor)
        self.reasoner = ClinicalReasonerAgent()
        self.atn = ATNBiomarkerProfiler(
            amyloid_threshold_cl=self.settings.amyloid_positive_centiloid,
            tau_threshold_suvr=self.settings.tau_positive_suvr,
        )
        pup_root = self.settings.pup_root
        if not os.path.isabs(pup_root):
            pup_root = os.path.join(ROOT, pup_root)
        self.pup = PUPPetParser(
            pup_root=pup_root,
            amyloid_threshold_cl=self.settings.amyloid_positive_centiloid,
            tau_threshold_suvr=self.settings.tau_positive_suvr,
        )
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def diagnose(self, req: dict) -> dict:
        pdata = req.get("patient_data", {})
        age = float(pdata.get("age", 0) or 0)
        mmse = float(pdata.get("mmse", 0) or 0)
        long_id = req.get("longitudinal_id")

        vision_result = {"class": "Not provided", "confidence": 0.0, "probabilities": []}
        explain = {"heatmap_available": False}
        if req.get("image_base64"):
            image = Image.open(io.BytesIO(base64.b64decode(req["image_base64"]))).convert("L")
            x = self.tf(image).unsqueeze(0).to(self.device)
            x.requires_grad = True
            out = self.model(x)
            probs = torch.softmax(out[0], dim=0)
            idx = int(torch.argmax(probs))
            vision_result = {
                "class": CLASS_NAMES[idx],
                "confidence": float(probs[idx] * 100),
                "probabilities": probs.detach().cpu().numpy().tolist(),
            }
            hm = self.explainer.generate_heatmap(x, target_class=idx)
            layers = render_gradcam(image, hm)
            explain = {"heatmap_available": True, "base_png": layers["base"],
                       "heatmap_png": layers["heatmap"], "overlay_png": layers["overlay"]}

        # Regional volumetry
        vol = self.volumetry.analyze_subject(long_id) if long_id else None
        if vol is None or vol.source == "unavailable":
            vol = self.volumetry.estimate_from_biomarkers(long_id or "demo", 1_500_000.0, 0.71)
        vol_dict = vol.to_dict()

        # ATN biomarker profile: A/T from OASIS-3 PET (PUP) when present, N from volumetry.
        pet = self.pup.analyze_subject(long_id) if long_id else None
        hippo_zs = [r.z_score for r in vol.regions if "Hippocampus" in r.structure]
        atn = self.atn.classify(
            amyloid_suvr=pet.amyloid_suvr if pet else None,
            amyloid_centiloid=pet.amyloid_centiloid if pet else None,
            amyloid_tracer=(pet.amyloid_tracer or "PIB") if pet else "PIB",
            tau_suvr=pet.tau_suvr if pet else None,
            hippocampus_z=(sum(hippo_zs) / len(hippo_zs)) if hippo_zs else None,
            mta_risk=vol.mta_risk_score,
        )
        atn_out = atn.to_dict()
        if pet:
            atn_out["pet"] = pet.to_dict()

        # Temporal (lightweight placeholder; full temporal lives in the FastAPI service)
        temporal = {"trend": "N/A (edge console)", "atrophy_velocity": 0.0}

        biomarker = {
            "age_risk": "elevated" if age > 70 else "normal",
            "mmse_category": ("severe_impairment" if mmse < 10 else "moderate_impairment" if mmse < 20
                              else "mild_impairment" if mmse < 24 else "normal"),
        }

        flagged, msg = self.ethicist.audit_diagnostic_proposal(
            vision_result["class"], vision_result["confidence"], mmse, temporal["atrophy_velocity"])
        final = vision_result["class"] if not flagged else "DIAGNOSIS_BLOCKED"

        reasoning = self.reasoner.synthesize({
            "prediction": vision_result["class"], "authorized_class": final,
            "confidence": vision_result["confidence"], "age": age, "mmse": mmse,
            "clinical_trend": temporal["trend"], "atrophy_velocity": 0.0,
            "volumetry_summary": vol.summary, "ethics_flagged": flagged,
            "ethics_message": msg, "rag_context": [],
        })

        return {
            "patient_id": pdata.get("patient_id", "UNKNOWN"),
            "vision_prediction": vision_result,
            "biomarker_analysis": biomarker,
            "temporal_analysis": temporal,
            "rag_context": [],
            "explainability": explain,
            "ethics_audit": {"approved": not flagged, "message": msg},
            "regional_volumetry": vol_dict,
            "atn_profile": atn_out,
            "clinical_narrative": reasoning.narrative,
            "reasoning_tier": f"{reasoning.tier}:{reasoning.model}",
            "final_diagnosis": final,
            "confidence": vision_result["confidence"],
            "approved": not flagged,
        }


ENGINE: "Engine | None" = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/app", "/console"):
            self.send_response(302)
            self.send_header("Location", "/app/")
            self.end_headers()
            return
        if path == "/health":
            return self._json(200, {"status": "healthy", "version": "edge-console", "device": str(ENGINE.device)})
        if path == "/models/info":
            return self._json(200, {"device": str(ENGINE.device),
                                    "acceleration": ENGINE.settings.onnx_providers,
                                    "llm": ENGINE.settings.summary()})
        if path == "/api/sample":
            return self._sample(parse_qs(parsed.query).get("label", [None])[0])
        if path.startswith("/app/"):
            return self._static(path[len("/app/"):] or "index.html")
        return self._json(404, {"detail": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/diagnose":
            return self._json(404, {"detail": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        try:
            return self._json(200, ENGINE.diagnose(req))
        except Exception as e:  # pragma: no cover
            return self._json(500, {"detail": str(e)})

    def _static(self, rel: str):
        safe = os.path.normpath(rel).lstrip("\\/")
        full = os.path.join(WEB, safe)
        if not full.startswith(WEB) or not os.path.isfile(full):
            return self._json(404, {"detail": "not found"})
        ext = os.path.splitext(full)[1].lower()
        with open(full, "rb") as fh:
            self._send(200, fh.read(), _MIME.get(ext, "application/octet-stream"))

    def _sample(self, label):
        data_root = os.path.join(ROOT, "data", "oasis_raw")
        dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        if label:
            dirs = [d for d in dirs if d.lower() == label.lower()] or dirs
        chosen = random.choice(dirs)
        cdir = os.path.join(data_root, chosen)
        imgs = [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        fname = random.choice(imgs)
        with open(os.path.join(cdir, fname), "rb") as fh:
            img = Image.open(io.BytesIO(fh.read())).convert("L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self._json(200, {"true_label": chosen, "filename": fname,
                         "image_base64": base64.b64encode(buf.getvalue()).decode()})


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print("[*] Initializing edge console engine (vision + Grad-CAM + volumetry + ethics + reasoner)…")
    ENGINE = Engine()
    print(f"[+] OASIS Clinical Console ready -> http://{args.host}:{args.port}/app/")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
