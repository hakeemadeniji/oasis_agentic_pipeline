"""
Patient-grouped (subject-disjoint) dataset splitting for the OASIS MRI slices.

WHY THIS EXISTS
---------------
The bundled OASIS slice dataset contains ~240 near-identical 2D slices per
subject. If train/validation/test are split at the *image* level, the same
patient's slices land in multiple splits and the model simply memorizes
subjects — producing wildly inflated, meaningless accuracy. Honest evaluation
requires that **no subject appears in more than one split**.

This module is the single source of truth for that split so the trainer and the
effectiveness-report evaluator agree on exactly which subjects are held out.
Splits are deterministic for a given ``(data_root, fractions, seed)``.

LIMITATION (bundled data): some classes have very few subjects (notably
"Moderate Dementia" has only 2). With subject-disjoint splitting such a class
can contribute at most a single held-out test patient, so its per-class metric
is statistically meaningless. This is a property of the data, not a bug — real
validation needs a many-subject cohort (OASIS-3 / ADNI). The split summary
exposes the per-class subject counts so reports can state this plainly.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from torchvision import datasets

# OASIS subject id, e.g. "OAS1_0028_MR1_mpr-1_100.jpg" -> "OAS1_0028".
# Different MR sessions (MR1/MR2) of one person share the subject id, so grouping
# on this correctly keeps all of a person's scans together.
_SUBJECT_RE = re.compile(r"(OAS\d+_\d+)", re.IGNORECASE)

Item = Tuple[str, int]  # (image_path, class_index)


def subject_id_from_path(path: str) -> str:
    """Extract the OASIS subject id from a slice path.

    Falls back to the filename stem (treating the file as its own "subject") when
    no OASIS id is present, so non-OASIS data still splits without crashing.
    """
    base = os.path.basename(path)
    m = _SUBJECT_RE.search(base)
    return m.group(1).upper() if m else os.path.splitext(base)[0]


def patient_grouped_split(
    data_root: str,
    *,
    test_frac: float = 0.2,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[Item], List[Item], List[Item]]:
    """Split an ImageFolder into subject-disjoint train/val/test sets.

    Stratified by each subject's (majority) class so every split sees every class
    that has enough subjects. Deterministic for a given (data_root, fracs, seed).

    Returns ``(classes, train_items, val_items, test_items)``.
    """
    ds = datasets.ImageFolder(root=data_root)
    classes = ds.classes

    by_subject: Dict[str, List[Item]] = defaultdict(list)
    for path, cls in ds.samples:
        by_subject[subject_id_from_path(path)].append((path, cls))

    # Each subject's label = the majority class of its slices (OASIS subjects are
    # single-class in practice, but this is robust if they are not).
    subj_label: Dict[str, int] = {}
    for subj, items in by_subject.items():
        labels = [c for _, c in items]
        subj_label[subj] = max(set(labels), key=labels.count)

    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[str]] = defaultdict(list)
    for subj in sorted(by_subject):  # sorted -> deterministic
        by_label[subj_label[subj]].append(subj)

    train_s: List[str] = []
    val_s: List[str] = []
    test_s: List[str] = []
    for _cls, subs in sorted(by_label.items()):
        subs = [subs[i] for i in rng.permutation(len(subs))]
        n = len(subs)
        # Reserve at least one test subject when a class has >1 subject; only
        # carve out a val subject when at least one would remain for training.
        n_test = max(1, round(n * test_frac)) if n > 1 else 0
        n_val = max(1, round(n * val_frac)) if (n - n_test) > 1 else 0
        test_s += subs[:n_test]
        val_s += subs[n_test : n_test + n_val]
        train_s += subs[n_test + n_val :]

    def gather(subjects: List[str]) -> List[Item]:
        items: List[Item] = []
        for subj in subjects:
            items += by_subject[subj]
        return items

    return classes, gather(train_s), gather(val_s), gather(test_s)


def split_summary(
    data_root: str, *, test_frac: float = 0.2, val_frac: float = 0.15, seed: int = 42
) -> Dict:
    """Subject + slice counts per class per split, for transparent reporting."""
    classes, tr, va, te = patient_grouped_split(
        data_root, test_frac=test_frac, val_frac=val_frac, seed=seed
    )

    def per_class_subjects(items: List[Item]) -> Dict[int, int]:
        subs: Dict[int, set] = defaultdict(set)
        for path, cls in items:
            subs[cls].add(subject_id_from_path(path))
        return {c: len(s) for c, s in subs.items()}

    return {
        "classes": classes,
        "train": {"slices": len(tr), "subjects": per_class_subjects(tr)},
        "val": {"slices": len(va), "subjects": per_class_subjects(va)},
        "test": {"slices": len(te), "subjects": per_class_subjects(te)},
    }


if __name__ == "__main__":
    import json

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "oasis_raw"))
    s = split_summary(root)
    print("Patient-grouped split (subject-disjoint) summary:")
    print("classes:", s["classes"])
    for part in ("train", "val", "test"):
        subs = s[part]["subjects"]
        print(
            f"  {part:5s}: {s[part]['slices']:6d} slices | subjects/class: "
            + json.dumps({s["classes"][c]: subs.get(c, 0) for c in range(len(s["classes"]))})
        )
