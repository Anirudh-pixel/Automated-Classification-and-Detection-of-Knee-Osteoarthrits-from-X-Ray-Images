"""
predict.py - Lean (ResNet50V2-only) cascade predictor for knee-OA X-ray screening.

EDUCATIONAL SCREENING DEMO. NOT A MEDICAL DIAGNOSIS.

Pipeline:
  Stage 1 (5-class KL grade 0-4) -> binary screen (Healthy vs Diseased)
  Stage 2 (2-class specialist)   -> Moderate vs Severe, only if Diseased

Serves ResNet50V2 only for each stage (no 6-model ensemble, no TTA) so it fits a
free CPU tier. Architecture is rebuilt in code and weights loaded with
load_weights(), so we never call load_model() and never touch WeightedFocalLoss.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF info logs

import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input

# ------------------------------- CONFIG -------------------------------------
IMG_SIZE = 320
STAGE1_CLASSES = 5
SPECIALIST_CLASSES = 2

# Decision thresholds. Recalibrated for the lean ResNet-only cascade (was 0.38/0.50
# on the full ensemble) to improve Severe recall - see calibrate_thresholds.py.
DISEASED_THRESHOLD = 0.32   # p_diseased >= this -> run the specialist
SEVERE_THRESHOLD = 0.48     # p_severe   >= this -> "Severe", else "Moderate"

# Non-X-ray guard. Real knee X-rays are pure grayscale (per-pixel R=G=B), so two
# cheap, independent checks gate the input. An image is rejected if EITHER trips:
#   1) mean per-pixel channel spread is too high (a real X-ray is ~0), or
#   2) too many pixels are visibly colored. Check 2 catches dim photos whose
#      AVERAGE saturation is low but which still contain colored patches (skin,
#      fabric, walls) - exactly the case that let a webcam selfie slip through.
XRAY_MEAN_SAT_MAX = 10.0          # reject if mean channel spread (0-255) exceeds this
XRAY_COLOR_PIXEL_SPREAD = 25      # a pixel is "colored" if its max-min channel > this
XRAY_COLOR_PIXEL_FRAC_MAX = 0.005 # reject if more than this fraction of pixels are colored

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", ".")
STAGE1_WEIGHTS = os.path.join(WEIGHTS_DIR, "final_g5_resnet.h5")
SPECIALIST_WEIGHTS = os.path.join(WEIGHTS_DIR, "spec_resnet_final.h5")

DISCLAIMER = (
    "EDUCATIONAL SCREENING DEMO - NOT A MEDICAL DIAGNOSIS. "
    "This is a student/research demonstration and may be wrong. It cannot replace "
    "a radiologist or clinician. If you have knee pain or any health concern, "
    "please consult a qualified medical professional."
)

GRADE_KEY = (
    "Grade key: Healthy = KL grade 0-1, Moderate = KL grade 2, "
    "Severe = KL grade 3-4."
)

# ----------------------------- MODEL BUILD ----------------------------------
def _build_model(num_classes, weights_path):
    """Rebuild the exact training architecture, then load_weights().

    backbone weights=None on purpose: load_weights() overwrites everything, and
    None stops the Space re-downloading ImageNet on every cold boot. The layer
    creation order MUST match training, because default load_weights() matches by
    topological order (not by name) -- if it mismatches it raises here at boot.
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Lambda(preprocess_input)(inputs)
    backbone = ResNet50V2(weights=None, include_top=False, input_tensor=x)
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs, name=f"resnet50v2_{num_classes}c")
    model.load_weights(weights_path)
    return model

# Build once at import so Gradio reuses the loaded models across requests.
print("Loading Stage-1 (5-class) ResNet50V2 ...")
_stage1 = _build_model(STAGE1_CLASSES, STAGE1_WEIGHTS)
print("Loading Specialist (2-class) ResNet50V2 ...")
_specialist = _build_model(SPECIALIST_CLASSES, SPECIALIST_WEIGHTS)
print("Models ready.")

# ----------------------------- PREPROCESS -----------------------------------
def _to_rgb_uint8(image):
    """Accept a PIL.Image, a file path, or a numpy array -> RGB uint8 array."""
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)

def _saturation_stats(rgb_uint8):
    """Return (mean_spread, colored_fraction) from the per-pixel channel spread.

    spread = max(R,G,B) - min(R,G,B) per pixel, on a 0-255 scale. A pure grayscale
    X-ray has spread 0 everywhere, so both stats are ~0. Photos -- even dim ones
    with a low average -- still have colored regions, so colored_fraction climbs.
    """
    arr = rgb_uint8.astype(np.int16)
    spread = arr.max(axis=2) - arr.min(axis=2)
    mean_spread = float(spread.mean())
    colored_fraction = float((spread > XRAY_COLOR_PIXEL_SPREAD).mean())
    return mean_spread, colored_fraction

def clahe_preprocess(rgb_uint8):
    """RGB -> resize(320) -> gray -> CLAHE -> back to 3-channel RGB float32 [0,255].

    NOTE: resize happens BEFORE CLAHE here. CLAHE is tile-based, so if your
    training applied CLAHE at the original resolution and resized afterwards, the
    output differs slightly -- tell me and I'll swap the order. The downstream
    Lambda(preprocess_input) does the ResNetV2 [-1,1] scaling, so we leave 0-255.
    """
    img = cv2.resize(rgb_uint8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32)

# ------------------------------- PREDICT ------------------------------------
def predict(image):
    """Run the lean cascade. Returns a dict with the verdict, probs, disclaimer."""
    if image is None:
        return {"error": "No image provided.", "disclaimer": DISCLAIMER}

    rgb = _to_rgb_uint8(image)

    # Non-X-ray guard runs on the RAW image, BEFORE CLAHE forces it to grayscale.
    mean_sat, colored_fraction = _saturation_stats(rgb)
    if mean_sat > XRAY_MEAN_SAT_MAX or colored_fraction > XRAY_COLOR_PIXEL_FRAC_MAX:
        return {
            "rejected": True,
            "reason": "This does not look like a grayscale X-ray, so no screening "
                      "was run. Please upload a knee X-ray image.",
            "channel_saturation": round(mean_sat, 1),
            "colored_fraction": round(colored_fraction, 4),
            "disclaimer": DISCLAIMER,
        }

    x = np.expand_dims(clahe_preprocess(rgb), axis=0)

    p5 = _stage1.predict(x, verbose=0)[0]
    p_diseased = float(p5[2] + p5[3] + p5[4])

    out = {
        "channel_saturation": round(mean_sat, 1),
        "colored_fraction": round(colored_fraction, 4),
        "grade_key": GRADE_KEY,
        "p_diseased": round(p_diseased, 4),
        "stage1_grade_probs": [round(float(v), 4) for v in p5],
        "disclaimer": DISCLAIMER,
    }

    if p_diseased < DISEASED_THRESHOLD:
        out["result"] = "Healthy"
        out["confidence"] = round(1.0 - p_diseased, 4)
        return out

    spec = _specialist.predict(x, verbose=0)[0]
    p_severe = float(spec[1])      # class 0 = Moderate, class 1 = Severe
    out["p_severe"] = round(p_severe, 4)
    out["result"] = "Severe" if p_severe >= SEVERE_THRESHOLD else "Moderate"
    out["confidence"] = round(
        p_severe if out["result"] == "Severe" else 1.0 - p_severe, 4
    )
    return out

if __name__ == "__main__":
    import sys
    from pprint import pprint
    if len(sys.argv) > 1:
        pprint(predict(sys.argv[1]))
    else:
        print("Usage: python predict.py path/to/xray.png")