from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from app.settings import (
    get_gemini_api_key,
    get_gemini_model,
    get_words_per_subtitle,
    has_gemini_api_key,
    set_gemini_api_key,
    set_gemini_model,
    set_words_per_subtitle,
)
from app.srt_pipeline import GeminiTranscriptionError, RECOMMENDED_TRANSCRIPTION_MODEL, generate_srt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


load_dotenv(ROOT / ".env")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_USERNAME = os.getenv("ALLOWED_TELEGRAM_USERNAME", "petfine").lstrip("@").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", RECOMMENDED_TRANSCRIPTION_MODEL)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", uuid.uuid4().hex)
MODEL_CHOICES = (
    RECOMMENDED_TRANSCRIPTION_MODEL,
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        await _deny(update)
        return
    await update.effective_message.reply_text(
        "ផ្ញើ audio ឬ video មកខ្ញុំ។ ខ្ញុំនឹងបង្កើត Khmer SRT ដែល timing ត្រូវតាមមាត់និយាយ។\n\n"
        "Commands:\n"
        "/menu - បើក menu\n"
        "/setgemini YOUR_KEY - ដាក់ Gemini API key\n"
        "/status - ពិនិត្យ setup"
    )
    await show_menu(update, context)


async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        await _deny(update)
        return

    await update.effective_message.reply_text(_settings_text(), reply_markup=_main_menu())


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
        f"Telegram: OK\n"
        f"Gemini key: {gemini_status}\n"
        f"Allowed user: @{ALLOWED_USERNAME}\n"
        f"Model: {get_gemini_model(GEMINI_MODEL)}\n"
        f"Words per subtitle: {get_words_per_subtitle()}"
    )


async def handle_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update):
        await _deny(update)
        return

    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data == "menu:api":
        await query.edit_message_text(
            "API menu\n\n"
            "Gemini key ត្រូវបានរក្សាទុក: "
            f"{'មានរួច' if has_gemini_api_key() else 'មិនទាន់មាន'}\n\n"
            "ដាក់ ឬប្តូរ Gemini API key:\n"
            "/setgemini YOUR_GEMINI_API_KEY",
            reply_markup=_main_menu(),
        )
        return

    if data == "menu:model":
        await query.edit_message_text(
            "ជ្រើស Model AI សម្រាប់ transcript សម្លេងទៅជា SRT timing:\n\n"
            f"Recommended: {RECOMMENDED_TRANSCRIPTION_MODEL}",
            reply_markup=_model_menu(),
        )
        return

    if data == "menu:words":
        await query.edit_message_text("ជ្រើសចំនួនពាក្យក្នុង subtitle មួយលោត:", reply_markup=_words_menu())
        return

    if data.startswith("model:"):
        model = data.split(":", 1)[1]
        if model in MODEL_CHOICES:
            set_gemini_model(model)
        await query.edit_message_text(_settings_text(), reply_markup=_main_menu())
        return

    if data.startswith("words:"):
        set_words_per_subtitle(int(data.split(":", 1)[1]))
        await query.edit_message_text(_settings_text(), reply_markup=_main_menu())
        return

    if data == "menu:back":
        await query.edit_message_text(_settings_text(), reply_markup=_main_menu())


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
            get_gemini_model(GEMINI_MODEL),
            get_words_per_subtitle(),
        )
    except FileNotFoundError as exc:
        await status.edit_text(f"រក command មិនឃើញ៖ {exc.filename}. សូមដំឡើង ffmpeg។")
        return
    except GeminiTranscriptionError as exc:
        await status.edit_text(exc.user_message)
        return
    except Exception as exc:
        await status.edit_text(
            "មានបញ្ហាពេលបង្កើត SRT។ នេះគឺជា bot transcript សម្លេង/video ទៅ SRT អក្សរលោតតាមមាត់និយាយ។ "
            f"សូមប្រើ model {RECOMMENDED_TRANSCRIPTION_MODEL} ក្នុង /menu ហើយសាកម្តងទៀត។\n\n"
            f"Error detail: {_short_error(exc)}"
        )
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


def _settings_text() -> str:
    return (
        "Fine SRT Menu\n\n"
        f"API: {'មានរួច' if has_gemini_api_key() else 'មិនទាន់មាន'}\n"
        f"Model AI: {get_gemini_model(GEMINI_MODEL)}\n"
        f"ពាក្យចេញម្តង: {get_words_per_subtitle()}"
    )


def _main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("API", callback_data="menu:api")],
            [InlineKeyboardButton("Model AI", callback_data="menu:model")],
            [InlineKeyboardButton("ពាក្យចេញ 1 / 2 / 3", callback_data="menu:words")],
        ]
    )


def _model_menu() -> InlineKeyboardMarkup:
    current = get_gemini_model(GEMINI_MODEL)
    rows = [
        [
            InlineKeyboardButton(
                f"{'✓ ' if model == current else ''}{model}{' ✅ សម្រាប់ SRT' if model == RECOMMENDED_TRANSCRIPTION_MODEL else ''}",
                callback_data=f"model:{model}",
            )
        ]
        for model in MODEL_CHOICES
    ]
    rows.append([InlineKeyboardButton("Back", callback_data="menu:back")])
    return InlineKeyboardMarkup(rows)


def _short_error(exc: Exception) -> str:
    text = str(exc)
    return text if len(text) <= 350 else text[:347] + "..."


def _words_menu() -> InlineKeyboardMarkup:
    current = get_words_per_subtitle()
    rows = [
        [
            InlineKeyboardButton(
                f"{'✓ ' if count == current else ''}{count}",
                callback_data=f"words:{count}",
            )
        ]
        for count in (1, 2, 3)
    ]
    rows.append([InlineKeyboardButton("Back", callback_data="menu:back")])
    return InlineKeyboardMarkup(rows)


async def _post_init(app: Application) -> None:
    await app.bot.set_my_commands(
        [
            BotCommand("start", "Start Fine SRT bot"),
            BotCommand("menu", "Open settings menu"),
            BotCommand("status", "Check API/model/subtitle settings"),
            BotCommand("setgemini", "Set Gemini API key"),
        ]
    )


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(_post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", show_menu))
    app.add_handler(CommandHandler("setgemini", setgemini))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CallbackQueryHandler(handle_menu_callback))
    app.add_handler(MessageHandler(filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Document.ALL, handle_media))

    if WEBHOOK_URL:
        webhook_path = WEBHOOK_SECRET
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
