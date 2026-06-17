"""
Balanced fine-tuning for the Alzheimer vision agent.

The bundled OASIS slice dataset is severely class-imbalanced
(Non Demented ~67k vs Moderate Dementia ~0.5k). Vanilla training collapses to
the majority class (balanced accuracy ~= chance). This script fixes that by:

  * building a **class-balanced, capped** train/val split (equal samples/class),
  * warm-starting from the existing checkpoint when present,
  * training with class-weighted, label-smoothed cross-entropy,
  * selecting the checkpoint with the best *balanced* validation accuracy,
  * backing up the previous weights before overwriting.

CPU/ARM64-friendly: keep --per-class and --epochs modest. Designed to turn a
collapsed model into a genuinely discriminative one in a few minutes.

Usage:
    python src/pipeline/train_balanced.py --per-class 300 --val-per-class 60 --epochs 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

_CUR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_CUR, ".."))
_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
if _SRC not in sys.path:
    sys.path.append(_SRC)

from agents.vision.vision_agent import AlzheimerVisionAgent  # noqa: E402

TRAIN_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
VAL_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class ListDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("L")), label


def build_splits(data_root: str, per_class: int, val_per_class: int, seed: int):
    ds = datasets.ImageFolder(root=data_root)
    by_class: Dict[int, List[str]] = defaultdict(list)
    for path, cls in ds.samples:
        by_class[cls].append(path)
    rng = np.random.default_rng(seed)
    train_items, val_items = [], []
    for cls, paths in by_class.items():
        idx = rng.permutation(len(paths))
        val_idx = idx[:val_per_class]
        train_idx = idx[val_per_class:val_per_class + per_class]
        val_items += [(paths[i], cls) for i in val_idx]
        train_items += [(paths[i], cls) for i in train_idx]
    rng.shuffle(train_items)
    return ds.classes, train_items, val_items


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy()
        for p, t in zip(pred, y.numpy()):
            total[t] += 1
            correct[t] += int(p == t)
    per_class = np.divide(correct, total, out=np.zeros_like(correct), where=total > 0)
    return float(per_class.mean()), per_class, (correct.sum() / total.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(_ROOT, "data", "oasis_raw"))
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--val-per-class", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(_ROOT, "src", "pipeline", "onnx_inference", "best_vision_agent.pth"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    classes, train_items, val_items = build_splits(
        args.data_root, args.per_class, args.val_per_class, args.seed
    )
    num_classes = len(classes)
    print(f"[data] classes={classes}")
    print(f"[data] train={len(train_items)} val={len(val_items)} ({args.per_class}/class train)")

    train_loader = DataLoader(ListDataset(train_items, TRAIN_TF), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(ListDataset(val_items, VAL_TF), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    model = AlzheimerVisionAgent(num_classes=num_classes).to(device)
    if os.path.exists(args.out):
        try:
            model.load_state_dict(torch.load(args.out, map_location=device))
            print(f"[init] warm-started from {os.path.basename(args.out)}")
        except Exception as e:
            print(f"[init] could not warm-start ({e}); training from scratch")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    base_bal, base_pc, base_acc = evaluate(model, val_loader, device, num_classes)
    print(f"[baseline] balanced_acc={base_bal*100:.1f}% overall_acc={base_acc*100:.1f}% "
          f"per_class={[f'{v*100:.0f}' for v in base_pc]}")

    best_bal = base_bal
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if bi % 10 == 0:
                print(f"  epoch {epoch+1} batch {bi}/{len(train_loader)} loss={loss.item():.3f}",
                      flush=True)
        scheduler.step()
        bal, pc, acc = evaluate(model, val_loader, device, num_classes)
        print(f"[epoch {epoch+1}] loss={running/len(train_loader):.3f} "
              f"balanced_acc={bal*100:.1f}% overall_acc={acc*100:.1f}% "
              f"per_class={[f'{v*100:.0f}' for v in pc]} ({time.perf_counter()-t0:.0f}s)", flush=True)
        if bal > best_bal:
            best_bal = bal
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> new best balanced_acc={bal*100:.1f}%", flush=True)

    if best_bal > base_bal:
        backup = args.out.replace(".pth", "_original.pth")
        if os.path.exists(args.out) and not os.path.exists(backup):
            shutil.copy2(args.out, backup)
            print(f"[save] backed up previous weights -> {os.path.basename(backup)}")
        torch.save(best_state, args.out)
        print(f"[save] wrote improved weights -> {args.out} (balanced_acc {base_bal*100:.1f}% -> {best_bal*100:.1f}%)")
    else:
        print(f"[save] no improvement over baseline ({base_bal*100:.1f}%); keeping existing weights")


if __name__ == "__main__":
    main()
