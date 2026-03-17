# AI_agents

# AI Meeting Intelligence Agent

A Flask-based “AI Meeting Intelligence” dashboard that turns meeting audio/transcripts into structured insights and follow-up workflows (Telegram, Email, Calendar). Includes a modern dark UI dashboard and a demo mode for presentations.

## Features
- **Audio → Insights pipeline (backend)**:
  - Local speech-to-text with Whisper
  - Optional speaker diarization with pyannote (fallback to single-speaker if gated/blocked)
  - Topic timeline segmentation with timestamps
  - Meeting intelligence extraction (summary, decisions, action items) using Groq
- **Automation workflows (backend)**:
  - Send meeting summary + action items to **Telegram**
  - Draft/send summary email via **SMTP**
  - Generate a calendar reminder JSON (ICS can be added later)
- **Reviewer Agent (Groq)**:
  - Audits action items vs speaker-attributed transcript and fixes owner/deadline mismatches before workflows execute
- **Dashboard (frontend)**:
  - Dark-themed UI with Summary, Timeline, Action Items table, Speaker Transcript, and a “Ask about this meeting” query bar
  - Lightweight `/auth` page (client-side) to capture user email for automatic delivery
- **Demo Mode (presentation)**:
  - Clicking “Process Audio” instantly fills the dashboard with realistic example content (no backend calls)

## Project Structure
- `app.py` — Flask API (audio processing, workflows, query endpoint, serves frontend)
- `transcriber.py` — Whisper + diarization + transcript attribution + topic segmentation
- `llm_client.py` — Groq Meeting Intelligence engine + Reviewer agent + Q&A helper
- `workflows.py` — Telegram, Email (SMTP), and Calendar reminder logic
- `segmentation.py` — Lightweight topic segmentation logic
- `meeting_models.py` — Pydantic models for `MeetingReport`
- `static/` — Frontend (`index.html`, `styles.css`, `script.js`, `auth.html`)

## API Endpoints
- `GET /` — Dashboard UI
- `GET /auth` — Login/signup page (stores email locally in browser)
- `GET /health` — Health check
- `POST /process_audio` — Upload audio file (`multipart/form-data` with `file`)
- `POST /execute_workflow` — Run workflow for `telegram` / `email` / `calendar`
- `POST /query_meeting` — Ask questions about the meeting (RAG-style Q&A)

## Setup (Windows)
Create a virtualenv and install deps:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Environment Variables
Required for Groq (meeting report + reviewer + query)
GROQ_API_KEY — Groq API key
GROQ_MODEL (optional) — defaults to llama-3.3-70b-versatile
Optional for speaker diarization (pyannote)
PYANNOTE_AUTH_TOKEN — Hugging Face token
PYANNOTE_DIARIZATION_MODEL (optional) — defaults to pyannote/speaker-diarization-3.1
DIARIZATION_ENABLED (optional) — set false to skip diarization
DIARIZATION_STRICT (optional) — set true to fail instead of fallback
Auto-email after processing audio (SMTP)
AUTO_EMAIL_ON_PROCESS_AUDIO=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=<your gmail>
SMTP_PASSWORD=<gmail app password>
EMAIL_FROM=<your gmail>
Note: Gmail requires a Google App Password (normal password won’t work).

Telegram (optional)
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
Run
venv\Scripts\activate
python app.py
Open:

http://127.0.0.1:5000/
