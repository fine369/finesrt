from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.settings import get_gemini_api_key, has_gemini_api_key, set_gemini_api_key
from app.srt_pipeline import generate_srt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


load_dotenv(ROOT / ".env")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_USERNAME = os.getenv("ALLOWED_TELEGRAM_USERNAME", "petfine").lstrip("@").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        await _deny(update)
        return
    await update.effective_message.reply_text(
        "ផ្ញើ audio ឬ video មកខ្ញុំ។ ខ្ញុំនឹងបង្កើត Khmer SRT ដែល timing ត្រូវតាមមាត់និយាយ។\n\n"
        "Commands:\n"
        "/setgemini YOUR_KEY - ដាក់ Gemini API key\n"
        "/status - ពិនិត្យ setup"
    )


async def setgemini(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        await _deny(update)
        return

    message = update.effective_message
    key = " ".join(context.args).strip()
    if not key:
        await message.reply_text("ប្រើបែបនេះ៖ /setgemini YOUR_GEMINI_API_KEY")
        return

    set_gemini_api_key(key)
    try:
        await message.delete()
    except Exception:
        pass
    await message.reply_text("Gemini API key បានរក្សាទុករួច។ ឥឡូវផ្ញើ audio/video មកបានហើយ។")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        await _deny(update)
        return

    gemini_status = "មានរួច" if has_gemini_api_key() else "មិនទាន់មាន"
    await update.effective_message.reply_text(
        f"Telegram: OK\nGemini key: {gemini_status}\nAllowed user: @{ALLOWED_USERNAME}\nModel: {GEMINI_MODEL}"
    )


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        await _deny(update)
        return

    message = update.effective_message
    media = message.video or message.audio or message.voice or message.document
    if not media:
        await message.reply_text("សូមផ្ញើជា audio/video file។")
        return

    if not has_gemini_api_key():
        await message.reply_text("សូមដាក់ Gemini key ជាមុនសិន៖ /setgemini YOUR_GEMINI_API_KEY")
        return

    size_mb = (media.file_size or 0) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        await message.reply_text(f"File ធំពេក។ កំណត់បច្ចុប្បន្នគឺ {MAX_UPLOAD_MB} MB។")
        return

    await message.chat.send_action(ChatAction.TYPING)
    status = await message.reply_text("កំពុងទាញយក file និងបង្កើត Khmer transcript...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tg_file = await media.get_file()
    suffix = Path(getattr(media, "file_name", "") or "").suffix or _suffix_for_media(update)
    input_path = DATA_DIR / f"{uuid.uuid4().hex}{suffix}"
    await tg_file.download_to_drive(custom_path=input_path)

    try:
        await status.edit_text("Gemini កំពុងស្តាប់ និងបំបែកជា segment មាន timestamp...")
        srt_path = await asyncio.to_thread(
            generate_srt,
            input_path,
            OUTPUT_DIR,
            get_gemini_api_key(),
            GEMINI_MODEL,
        )
    except FileNotFoundError as exc:
        await status.edit_text(f"រក command មិនឃើញ៖ {exc.filename}. សូមដំឡើង ffmpeg។")
        return
    except Exception as exc:
        await status.edit_text(f"មានបញ្ហាពេលបង្កើត SRT: {exc}")
        return

    await status.edit_text("រួចរាល់។ នេះជា Khmer SRT។")
    await message.reply_document(document=srt_path.open("rb"), filename=srt_path.name)


def _allowed(update: Update) -> bool:
    user = update.effective_user
    if not user or not user.username:
        return False
    return user.username.lower() == ALLOWED_USERNAME


async def _deny(update: Update) -> None:
    await update.effective_message.reply_text("Bot នេះអនុញ្ញាតតែ Telegram @petfine ប៉ុណ្ណោះ។")


def _suffix_for_media(update: Update) -> str:
    message = update.effective_message
    if message.voice:
        return ".ogg"
    if message.audio:
        return ".mp3"
    if message.video:
        return ".mp4"
    return ".bin"


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("setgemini", setgemini))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Document.ALL, handle_media))

    if WEBHOOK_URL:
        webhook_path = TELEGRAM_BOT_TOKEN
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=webhook_path,
            webhook_url=f"{WEBHOOK_URL}/{webhook_path}",
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
