"""
discord_bot.py - Discord bot for the lean KOA X-ray screening cascade.

Runs LOCALLY (e.g. in your tf_gpu env) and imports predict.py directly, so it
reuses the same models, weights, and logic as the Space and the Telegram bot.
It connects over Discord's websocket gateway, so it works while this script runs -
no public web server needed (just like the Telegram bot).

Setup (see chat for click-by-click):
  1. Create an application + bot at https://discord.com/developers/applications
  2. On the Bot page: enable the "MESSAGE CONTENT INTENT" toggle, and copy the token.
  3. Invite the bot to a server via OAuth2 -> URL Generator (scope: bot).
  4. pip install -U discord.py
  5. Provide the token (environment variable preferred):
       PowerShell:  $env:DISCORD_BOT_TOKEN = "your-token"
  6. Run from your KOA folder (next to predict.py + the .h5 files):
       python discord_bot.py

Usage: DM the bot a knee X-ray image, OR in a server @mention the bot with an
image attached. (Restricting to DMs/mentions keeps it from replying to every
image posted in a busy channel.)
"""

import os
from io import BytesIO

import discord
from PIL import Image

from predict import predict, DISCLAIMER, GRADE_KEY  # loads both models once, at import

TOKEN = "YOUR_DISCORD_TOKEN"

WELCOME = (
    "Knee OA X-ray Screening Bot (educational demo).\n"
    "DM me a knee X-ray image, or @mention me with one attached, and I'll screen it.\n\n"
    + DISCLAIMER
)

# message_content is a PRIVILEGED intent: this code line is necessary but NOT
# sufficient - you must ALSO toggle it ON in the Developer Portal (Bot page).
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


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


def first_image_attachment(message):
    for att in message.attachments:
        if att.content_type and att.content_type.startswith("image/"):
            return att
    return None


@client.event
async def on_ready():
    print(f"Logged in as {client.user} - bot is running.")


@client.event
async def on_message(message):
    if message.author == client.user:
        return  # never react to our own messages

    is_dm = message.guild is None
    mentioned = client.user in message.mentions
    if not (is_dm or mentioned):
        return  # in servers, only respond when explicitly @mentioned

    att = first_image_attachment(message)
    if att is None:
        await message.channel.send(WELCOME)
        return

    data = await att.read()
    try:
        image = Image.open(BytesIO(data))
    except Exception:
        await message.channel.send("I couldn't read that as an image. Send a PNG/JPG X-ray.")
        return

    # predict() blocks briefly during inference. On your GPU machine that's fast;
    # if you ever see "heartbeat blocked" warnings, wrap it in asyncio.to_thread.
    async with message.channel.typing():
        result = predict(image)
    await message.channel.send(format_result(result))


def main():
    if TOKEN == "PASTE_YOUR_TOKEN_HERE":
        raise SystemExit(
            "No token set. Set the DISCORD_BOT_TOKEN environment variable, or paste "
            "your bot token into the TOKEN line in discord_bot.py."
        )
    client.run(TOKEN)


if __name__ == "__main__":
    main()