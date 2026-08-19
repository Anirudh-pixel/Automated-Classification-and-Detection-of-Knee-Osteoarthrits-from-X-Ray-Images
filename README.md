---
title: Knee OA X-ray Screening (Educational Demo)
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.17.3
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Knee Osteoarthritis X-ray Screening — Educational Demo

**EDUCATIONAL SCREENING DEMO — NOT A MEDICAL DIAGNOSIS.**
This is a student/research demonstration and may be wrong. It cannot replace a
radiologist or clinician. If you have knee pain or any health concern, please
consult a qualified medical professional.

## What this does

Upload a knee X-ray. A two-stage cascade runs:

1. **Stage 1 (screening):** a ResNet50V2 5-class model (KL grades 0–4) whose
   probabilities are folded into Healthy vs Diseased.
2. **Stage 2 (specialist):** if Diseased, a second ResNet50V2 (2-class) splits
   Moderate vs Severe.

For speed on the free CPU tier this live demo serves a single ResNet50V2 per
stage (no full ensemble, no test-time augmentation). Inputs that are not roughly
grayscale (i.e. not X-ray-like) are rejected before any prediction is made.
