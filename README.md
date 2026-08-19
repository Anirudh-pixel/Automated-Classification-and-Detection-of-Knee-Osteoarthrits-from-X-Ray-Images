# KOA Screen — Knee Osteoarthritis X-ray Severity Screening

> **Educational / research demonstration — not a medical diagnosis.** This project is a student/research demo and may be wrong. It cannot replace a radiologist or clinician. If you have knee pain or any health concern, consult a qualified medical professional.

A two-stage deep-learning system that reads a knee X-ray and estimates osteoarthritis severity on the **Kellgren–Lawrence (KL)** scale, returning a simple verdict — **Healthy**, **Moderate**, or **Severe** — with a confidence score. The same model is deployed across a web demo, three chat channels, and a website.

- **Live web demo:** https://huggingface.co/spaces/Anirudh2474/knee-oa-xray-demo
- **Telegram bot:** https://t.me/koa_xray_demo_bot
- **Discord bot:** [add to a server](https://discord.com/oauth2/authorize?client_id=1514886352502526114&permissions=101376&integration_type=0&scope=bot)
- **WhatsApp:** https://wa.me/15556762167 *(test recipients only, for now)*

<!-- TODO: fill these in -->
*Author:* `____________`  ·  *Guide:* `____________`  ·  *Repo:* `https://github.com/<your-username>/<repo-name>`

---

## What it does

1. A knee radiograph is uploaded.
2. A **safety guard** rejects anything that isn't a grayscale X-ray (color photos, selfies, etc.).
3. The image is preprocessed (**CLAHE** contrast enhancement, resized to 320×320).
4. A **two-stage cascade** grades it and returns the verdict, confidence, and a disclaimer.

**Grade mapping:** `Healthy = KL 0–1` · `Moderate = KL 2` · `Severe = KL 3–4`

> Note on naming: public datasets label the KL-3 folder "Moderate" and KL-4 "Severe". In this project KL-3 is treated as **Severe** (grades 3–4), so a dataset image under `3Moderate` is a *Severe* case here. Mapping is done by KL grade number, not the folder word.

---

## Model architecture

A cascade of two models — screen first, then specialise:

- **Stage 1 — Screen (Healthy vs Diseased).** A 5-class KL model; the diseased probability is `p(2) + p(3) + p(4)`. If it is below the **diseased threshold (0.32)**, the result is **Healthy**.
- **Stage 2 — Specialist (Moderate vs Severe).** Runs only on diseased knees. If `p(severe)` is at or above the **severe threshold (0.48)**, the result is **Severe**, else **Moderate**.

**Two builds:**
- **Full model (report):** an ensemble of **EfficientNetB0 + DenseNet121 + ResNet50V2** per stage, with test-time augmentation (TTA).
- **Lean model (served live):** a single **ResNet50V2** per stage, no TTA — light enough to run on a free CPU server.

Preprocessing: grayscale → CLAHE (clipLimit 2.0, 8×8 tiles) → 3-channel → `resnet_v2.preprocess_input`. Models are rebuilt in code and loaded via `load_weights()` (not `load_model`).

---

## Results

Evaluated on **MedicalExpert-II (1,650 images)** — a *cross-domain* set the model never trained on.

| Build | Task | Metric | Score |
|-------|------|--------|-------|
| Full ensemble | Healthy vs Diseased | Accuracy | ~90% |
| Full ensemble | 3-class (H/M/S) | Accuracy | ~83% |
| Lean (served) | 3-class (H/M/S) | Balanced accuracy | ~67% |
| Lean (served) | Severe recall | Recall | ~70% (up from 58% after threshold recalibration) |

Recalibrating the lean model's thresholds from 0.38/0.50 to **0.32/0.48** raised Severe recall by ~12 points (it was under-calling severe cases) at a small cost to Healthy recall — the safer direction for a screening tool.

---

## Datasets

The datasets and trained weights are **not stored in this repo** (they are large). Download the datasets from the sources below. Folder names in the project map to them as follows:

| Project folder | Dataset | Link |
|----------------|---------|------|
| `kaggle_dataset_2018/` | Knee Osteoarthritis Dataset with KL Grading (2018) | https://www.kaggle.com/datasets/tommyngx/kneeoa |
| `kaggle_dataset2_2020/` | Digital Knee X-ray Images (2020) — contains **MedicalExpert-I / MedicalExpert-II** | https://www.kaggle.com/datasets/tommyngx/digital-knee-xray |
| *(cross-domain test)* | **MedicalExpert-II** — the unseen test set for evaluation | part of the Digital Knee X-ray Images dataset above |
| `graded_dataset_5/`, `graded_extra_test_2020_E1/`, `stage1_binary/`, `stage2_mod_vs_sev/` | Processed / re-organised subsets **derived** from the datasets above | — (generated locally) |

**Original sources & citations (for the report):**

- **Knee Osteoarthritis Severity Grading** (OAI-organised, KL 0–4) — also mirrored on Kaggle by Shashwat Tiwari: https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity · Mendeley original (Chen, 2018): https://data.mendeley.com/datasets/56rmx5bjcr/1 (DOI: 10.17632/56rmx5bjcr.1)
- **Digital Knee X-ray Images (MedicalExpert-I / II)** — Gornale, S., & Patravali, P. (2020). *Digital Knee X-ray Images.* Mendeley Data, V1. https://data.mendeley.com/datasets/t9ndx37v5h/1 (DOI: 10.17632/t9ndx37v5h.1)

> If you downloaded a different mirror of these datasets, swap the link above to match. These are the standard public sources matching the folder names used here.

---

## Repository structure

```
KOA/
├─ predict.py               # core: guard + two-stage cascade + thresholds
├─ app.py                   # Gradio web app (Hugging Face Space)
├─ bot.py                   # Telegram bot
├─ discord_bot.py           # Discord bot
├─ app_whatsapp.py          # WhatsApp webhook (FastAPI + Meta Cloud API)
├─ calibrate_thresholds.py  # threshold recalibration script
├─ index.html               # showcase website (embeds the live Space)
├─ requirements.txt
├─ README.md
└─ <training notebook>.ipynb # model training / experiments
```

**Not in the repo (by design):** the `*.h5` weight files and the dataset image folders — see `.gitignore`.

---

## Setup & running locally

**1. Clone and create an environment** (Python 3.10; the models were trained on TensorFlow 2.10):

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**2. Get the model weights.** The trained weights live on the Hugging Face Space. Download `final_g5_resnet.h5` and `spec_resnet_final.h5` from the Space's **Files** tab and place them in the project root:

https://huggingface.co/spaces/Anirudh2474/knee-oa-xray-demo/tree/main

**3. Run the web app locally:**

```bash
python app.py
```

**4. Run a bot** (tokens are read from environment variables — never hard-code them):

```bash
# example (PowerShell)
$env:TELEGRAM_BOT_TOKEN="your-token-here"
python bot.py
```

Bots reply only while their script is running. The Hugging Face Space is always on (first request after idle can take 30–60s to wake).

---

## Security notes

- **Never commit secrets.** Bot tokens and the WhatsApp access token / `VERIFY_TOKEN` are read from environment variables and are excluded by `.gitignore`. If a token is ever committed, rotate it immediately.
- The Discord **Client ID** in `index.html` is public and safe.
- Model weights are hosted on the Hugging Face Space, not in git.

---

## Disclaimer

This is an **educational and research demonstration only**. It is **not a medical device**, has **not been clinically validated**, and must **not** be used for diagnosis or treatment decisions. The live demo serves a single ResNet50V2 per stage for speed; accuracy is lower than the full ensemble reported above. Always consult a qualified medical professional for any health concern.
