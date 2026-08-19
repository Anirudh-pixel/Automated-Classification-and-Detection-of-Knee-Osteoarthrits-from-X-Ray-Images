"""
app_whatsapp.py - WhatsApp (Meta Cloud API) webhook for the KOA X-ray cascade.

Architecture (WhatsApp PUSHES to you, unlike Telegram/Discord):
  WhatsApp user -> Meta Cloud API -> public HTTPS (ngrok) -> THIS FastAPI server
  -> predict() -> reply sent back via the Meta Graph API.

Env vars required (set them in the same terminal before running):
  WHATSAPP_TOKEN   - access token from the Meta App "API Setup" panel (temp = ~24h!)
  PHONE_NUMBER_ID  - the test sender's "Phone number ID" (same panel)
  VERIFY_TOKEN     - ANY string you invent; paste the SAME string into Meta's webhook

Run from your KOA folder (next to predict.py + the .h5 files):
  pip install fastapi uvicorn requests
  uvicorn app_whatsapp:app --port 8000
Then in a second terminal expose it publicly:
  ngrok http 8000
Your webhook Callback URL is the ngrok https URL + "/webhook".
"""

import os
from io import BytesIO

import requests
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from predict import predict, DISCLAIMER, GRADE_KEY  # loads both models once, at import

# If Meta's API-Setup sample code shows a different version, change this to match.
GRAPH_VERSION = "v22.0"
GRAPH = f"https://graph.facebook.com/{GRAPH_VERSION}"

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID", "")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "")

WELCOME = (
    "Knee OA X-ray Screening Bot (educational demo). "
    "Send me a knee X-ray IMAGE and I'll screen it.\n\n" + DISCLAIMER
)

app = FastAPI()


def format_result(result: dict) -> str:
    if result.get("rejected"):
        return "NOT AN X-RAY - no screening was run.\n\n" + result["reason"]
    if "error" in result:
        return result["error"]
    verdict = result["result"]
    conf = result.get("confidence", 0.0)
    lines = [
        f"Screening result: {verdict}",
        f"Confidence: {conf:.0%}",
        "",
        f"Diseased probability: {result['p_diseased']:.0%}",
    ]
    if "p_severe" in result:
        lines.append(f"Severe probability: {result['p_severe']:.0%}")
    lines += ["", GRADE_KEY, "", result["disclaimer"]]
    return "\n".join(lines)


def download_media(media_id: str) -> bytes:
    """Meta sends only a media_id. Resolve it to a URL, then fetch the bytes.
    BOTH calls need the bearer token."""
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    meta = requests.get(f"{GRAPH}/{media_id}", headers=headers, timeout=30).json()
    return requests.get(meta["url"], headers=headers, timeout=30).content


def send_text(to_number: str, body: str) -> None:
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": body[:4000]},   # WhatsApp text body limit is ~4096 chars
    }
    r = requests.post(f"{GRAPH}/{PHONE_NUMBER_ID}/messages",headers=headers, json=payload, timeout=30)
    if r.status_code >= 400:
        print("send_text error:", r.status_code, r.text)


def handle_message(msg: dict, from_number: str) -> None:
    """Runs in a worker thread (offloaded), so blocking I/O + inference is fine."""
    if msg.get("type") == "image":
        try:
            data = download_media(msg["image"]["id"])
            image = Image.open(BytesIO(data))
        except Exception as e:
            print("media error:", e)
            send_text(from_number, "I couldn't read that image. Send a PNG/JPG X-ray.")
            return
        result = predict(image)
        send_text(from_number, format_result(result))
    else:
        send_text(from_number, WELCOME)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/webhook")
def verify(request: Request):
    """Meta's one-time verification handshake (GET)."""
    p = request.query_params
    if p.get("hub.mode") == "subscribe" and p.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(p.get("hub.challenge", ""))
    return PlainTextResponse("Forbidden", status_code=403)


@app.post("/webhook")
async def incoming(request: Request):
    """Incoming events (POST). Return 200 quickly; do heavy work off the loop."""
    data = await request.json()
    try:
        value = data["entry"][0]["changes"][0]["value"]
        messages = value.get("messages")
        if messages:                      # ignore delivery/read status callbacks
            msg = messages[0]
            await run_in_threadpool(handle_message, msg, msg["from"])
    except Exception as e:
        print("webhook error:", e)
    return Response(status_code=200)