"""
bot.py - Telegram bot for the lean KOA X-ray screening cascade.

Runs LOCALLY (e.g. in your tf_gpu env) and imports predict.py directly, so it
reuses the exact same models, weights, and decision logic as the Space.
The bot only responds while this script is running on your machine.

Setup (see the chat for click-by-click):
  1. Create a bot via @BotFather on Telegram and copy the token it gives you.
  2. Provide the token, preferably as an environment variable:
       PowerShell:  $env:TELEGRAM_BOT_TOKEN = "123456789:ABC-your-token"
     (or paste it into TOKEN below - less safe, never share/commit it).
  3. Install the library:  pip install "python-telegram-bot>=21,<23"
  4. Run from your KOA folder (next to predict.py + the .h5 files):  python bot.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # run on CPU: lets several bots run at once
from io import BytesIO

from PIL import Image
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from predict import predict, DISCLAIMER, GRADE_KEY  # loads both models once, at import

# Prefer the environment variable; fall back to a pasted token.
TOKEN ="YOUR_TELEGRAM_TOKEN"

WELCOME = (
    "Knee OA X-ray Screening Bot (educational demo).\n\n"
    "Send me a knee X-ray image (as a photo or an image file) and I'll screen it.\n\n"
    + DISCLAIMER
)


def format_result(result: dict) -> str:
    """Turn the predict() dict into a readable Telegram message."""
    if result.get("rejected"):
        return (
            "NOT AN X-RAY - no screening was run.\n\n"
            f"{result['reason']}\n\n"
            f"{result['disclaimer']}"
        )
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message

    if msg.photo:                                  # sent as a compressed photo
        tg_file = await msg.photo[-1].get_file()
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        tg_file = await msg.document.get_file()    # sent as an image file
    else:
        await msg.reply_text("Please send a knee X-ray as a photo or image file.")
        return

    buf = BytesIO()
    await tg_file.download_to_memory(buf)
    buf.seek(0)
    image = Image.open(buf)

    # predict() is synchronous and blocks for a few seconds during inference.
    # Fine for a demo bot; for heavy concurrency wrap it in asyncio.to_thread.
    result = predict(image)
    await msg.reply_text(format_result(result))


def main() -> None:
    if TOKEN == "PASTE_YOUR_TOKEN_HERE":
        raise SystemExit(
            "No token set. Set the TELEGRAM_BOT_TOKEN environment variable, or "
            "paste your BotFather token into the TOKEN line in bot.py."
        )
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, start))
    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()