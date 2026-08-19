"""
calibrate_thresholds.py - Re-tune the LEAN (ResNet-only) cascade's two decision
thresholds on labeled data, optimizing BALANCED 3-class accuracy.

WHY: the 0.38 / 0.50 thresholds in predict.py were tuned on the FULL 6-model
ensemble + TTA. The single ResNet served live behaves differently (it has been
under-calling Severe), so its thresholds should be re-fit to its own outputs.

Run from your KOA folder (so it reuses the SAME models/weights as predict.py):
    python calibrate_thresholds.py

IMPORTANT naming note: the dataset folders use standard KL names, where
"3Moderate" = KL grade 3 and "4Severe" = KL grade 4. Your MODEL's 3-class scheme
is different: Healthy = KL 0-1, Moderate = KL 2, Severe = KL 3-4. This script maps
each folder by its KL GRADE NUMBER (leading digit), then converts to your scheme -
so a "3Moderate" image counts as a true *Severe*. That is correct and matches your
cascade.
"""

import os
import re
import glob

import numpy as np
from PIL import Image

# Reuse the EXACT models + preprocessing the live Space/bot use.
from predict import _stage1, _specialist, clahe_preprocess, _to_rgb_uint8

# ---- point this at your MedicalExpert-II folder (relative to KOA) ----
DATA_DIR = os.path.join("kaggle_dataset2_2020", "MedicalExpert-II")

BATCH = 16
CLASSES = ["Healthy", "Moderate", "Severe"]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Fallback ONLY if a folder name has no leading digit. Maps DATASET names to KL
# grade numbers (NOT to your 3-class labels). Note "moderate"->KL3, "severe"->KL4.
NAME_TO_KL = {
    "normal": 0, "healthy": 0,
    "doubtful": 1,
    "minimal": 2, "mild": 2,
    "moderate": 3,
    "severe": 4,
}


def folder_to_kl(folder_name):
    m = re.match(r"\s*(\d)", folder_name)
    if m:
        return int(m.group(1))
    low = folder_name.lower()
    for key, kl in NAME_TO_KL.items():
        if key in low:
            return kl
    return None


def kl_to_3class(kl):
    """Your model's scheme: Healthy = KL 0-1, Moderate = KL 2, Severe = KL 3-4."""
    if kl in (0, 1):
        return "Healthy"
    if kl == 2:
        return "Moderate"
    return "Severe"


def gather_images(data_dir):
    items = []  # (true_3class, path)
    print(f"Scanning {data_dir} ...")
    for folder in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, folder)
        if not os.path.isdir(fpath):
            continue
        kl = folder_to_kl(folder)
        if kl is None:
            print(f"  ! Skipping '{folder}' - could not resolve a KL grade.")
            continue
        true3 = kl_to_3class(kl)
        paths = [p for p in glob.glob(os.path.join(fpath, "*"))
                 if p.lower().endswith(IMG_EXTS)]
        print(f"  {folder:<16} -> KL {kl} -> {true3:<8} ({len(paths)} images)")
        for p in paths:
            items.append((true3, p))
    return items


def run_models(items):
    trues, p_dis, p_sev = [], [], []
    n = len(items)
    for i in range(0, n, BATCH):
        chunk = items[i:i + BATCH]
        batch_imgs, batch_true = [], []
        for true3, path in chunk:
            try:
                batch_imgs.append(clahe_preprocess(_to_rgb_uint8(Image.open(path))))
                batch_true.append(true3)
            except Exception as e:
                print(f"\n  ! Skipping unreadable file {path}: {e}")
        if not batch_imgs:
            continue
        x = np.stack(batch_imgs)
        p5 = _stage1(x, training=False).numpy()
        spec = _specialist(x, training=False).numpy()
        for j, true3 in enumerate(batch_true):
            trues.append(true3)
            p_dis.append(float(p5[j, 2] + p5[j, 3] + p5[j, 4]))
            p_sev.append(float(spec[j, 1]))
        print(f"  processed {min(i + BATCH, n)}/{n}", end="\r")
    print()
    return np.array(trues), np.array(p_dis), np.array(p_sev)


def predict_labels(p_dis, p_sev, t_dis, t_sev):
    return np.where(p_dis < t_dis, "Healthy",
                    np.where(p_sev >= t_sev, "Severe", "Moderate"))


def metrics(trues, preds):
    """Return (overall_acc, balanced_acc, per_class_recall, confusion_matrix).

    Balanced accuracy averages recall only over classes that actually appear.
    """
    idx = {c: k for k, c in enumerate(CLASSES)}
    mat = np.zeros((3, 3), dtype=int)
    for t, p in zip(trues, preds):
        mat[idx[t], idx[p]] += 1
    plain = float((preds == trues).mean())
    recalls, present = [], []
    for k in range(3):
        total = mat[k].sum()
        if total:
            recalls.append(mat[k, k] / total)
            present.append(recalls[-1])
        else:
            recalls.append(None)
    balanced = float(np.mean(present)) if present else 0.0
    return plain, balanced, recalls, mat


def print_report(title, trues, preds):
    plain, balanced, recalls, mat = metrics(trues, preds)
    print(f"\n=== {title} ===")
    print(f"  overall accuracy : {plain:.3f}")
    print(f"  balanced accuracy: {balanced:.3f}")
    for c, r in zip(CLASSES, recalls):
        print(f"    recall {c:<8}: {'n/a' if r is None else f'{r:.3f}'}")
    print("  confusion (rows=true, cols=pred)  [Healthy, Moderate, Severe]")
    for c, row in zip(CLASSES, mat):
        print(f"    {c:<8} {row}")


def main():
    items = gather_images(DATA_DIR)
    if not items:
        raise SystemExit(f"No labeled images found under {DATA_DIR}. Check the path.")

    print(f"\nRunning the lean pipeline on {len(items)} images "
          f"(uses your GPU; may take a few minutes)...")
    trues, p_dis, p_sev = run_models(items)

    print_report("CURRENT thresholds (diseased=0.38, severe=0.50)",
                 trues, predict_labels(p_dis, p_sev, 0.38, 0.50))

    # Grid search; maximize balanced accuracy, tie-break by overall accuracy.
    best = None
    for t_dis in np.round(np.arange(0.20, 0.601, 0.01), 2):
        for t_sev in np.round(np.arange(0.20, 0.801, 0.01), 2):
            preds = predict_labels(p_dis, p_sev, t_dis, t_sev)
            plain, balanced, _, _ = metrics(trues, preds)
            score = (balanced, plain)
            if best is None or score > best[0]:
                best = (score, float(t_dis), float(t_sev))

    (bal, plain), t_dis, t_sev = best
    print_report(f"BEST thresholds (diseased={t_dis:.2f}, severe={t_sev:.2f})",
                 trues, predict_labels(p_dis, p_sev, t_dis, t_sev))

    print("\n" + "=" * 60)
    print("To deploy these, edit predict.py and set:")
    print(f"    DISEASED_THRESHOLD = {t_dis:.2f}")
    print(f"    SEVERE_THRESHOLD   = {t_sev:.2f}")
    print("then re-upload predict.py to the Space, and restart your bot.")
    print("=" * 60)
    print("\nNote: thresholds were tuned and scored on the same set, so this")
    print("balanced accuracy is slightly optimistic. See the chat for a one-line")
    print("split if you want a cleaner number to report.")


if __name__ == "__main__":
    main()