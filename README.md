# Khmer SRT Telegram Bot

Telegram bot for one allowed user, `@petfine`, that turns audio/video into Khmer `.srt` subtitles:

`Gemini -> Khmer transcript -> segment alignment -> timing correction -> SRT`

## Setup

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

On this Codex machine you can also run the helper script:

```powershell
.\run_bot.ps1
```

2. Copy `.env.example` to `.env` and fill in:

```powershell
Copy-Item .env.example .env
```

Required values:

- `TELEGRAM_BOT_TOKEN`
- `GEMINI_API_KEY`
- `ALLOWED_TELEGRAM_USERNAME=petfine`
- `WEBHOOK_URL` only on Render, for example `https://khmer-srt-telegram-bot.onrender.com`
- `WEBHOOK_SECRET` only on Render, any random secret string

3. Install `ffmpeg` if you want to process video files. Audio files can work directly, but video-to-audio extraction needs `ffmpeg`.

4. Run:

```powershell
.\.venv\Scripts\python -m app.bot
```

## Usage

Open the bot in Telegram as `@petfine`, send `/start`, then set your Gemini key:

```text
/setgemini YOUR_GEMINI_API_KEY
```

After that, upload an audio or video file. The bot replies with a Khmer/English `.srt` file. Subtitles are split into short word beats, so each space-separated spoken unit appears as its own timed subtitle.

Use `/status` to check whether the Gemini key is configured.

## Free Server Deploy

This project includes a `Dockerfile` and `render.yaml`.

For Render:

1. Push this folder to a GitHub repository.
2. Create a new Render service from the repository.
3. Choose Docker.
4. Add environment variables:
   - `TELEGRAM_BOT_TOKEN`
   - `GEMINI_API_KEY`
   - `WEBHOOK_URL`, for example `https://khmer-srt-telegram-bot.onrender.com`
   - `WEBHOOK_SECRET`, or let Render generate it from `render.yaml`
   - `ALLOWED_TELEGRAM_USERNAME=petfine`
5. Deploy.

Important: this uses a Render free web service with a Telegram webhook. Render free services can still spin down when idle and have monthly free-hour limits, so the first message after idle can be delayed.

## Notes

- `.env`, uploaded media, generated outputs, and bot-stored `data/settings.json` are ignored by git.
- Timing quality depends on Gemini timestamp quality. The app applies a correction pass to avoid overlapping subtitle blocks, keep short readable durations, and clamp timings to media duration when `ffprobe` is available.
