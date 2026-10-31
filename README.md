# Interview Sparring Partner

A voice-based AI mock-interview app. Upload your resume (PDF), and it generates personalized
interview questions, speaks each question aloud in Hinglish, listens to your spoken answer,
transcribes it, and gives you per-answer feedback plus an end-of-session report — like a
sparring partner before real placements.

**Live demo:** https://interview-sparring-partner.onrender.com (free tier — the first load
after a quiet spell can take about a minute while the service wakes up.) The full voice loop
works on the live demo too: there, transcription runs on Groq's hosted Whisper API
(`ISP_TRANSCRIBE=groq`) instead of a local whisper.cpp server.

## Why I built this

I'm in my final year, and placement season made one thing clear: solving problems on
paper and *speaking* an answer out loud — structured, without a dozen fillers — are two
different skills, and almost nobody around me had time to take my mock interviews.
So I built the partner I needed. It reads my own resume, asks in the Hinglish that real
Indian interviewers actually use, and keeps score like a strict judge.

## Known rough edges

- Locally, the whisper.cpp server has to be started separately before the app
  is useful (the hosted demo sidesteps this with Groq's Whisper API).
  A one-command startup is on my list.
- Scores wobble slightly between runs — the evaluator is an LLM, after all. The
  filler-word count, at least, is exact: it is counted in Python, never by the LLM.
- Questions are generated from the parsed resume, so an odd PDF layout can mean a
  weaker parse and more generic questions. Common resume layouts work best.

## Architecture

```
Browser (web/index.html + app.js + style.css, vanilla JS, no frameworks)
  |  fetch JSON / multipart / audio
  v
FastAPI (ispar/app.py)  ---- serves web/ statically at /
  |
  |-- resume parse: pdfplumber
  |-- speech-to-text: whisper.cpp HTTP server on :8080
  |-- text-to-speech: edge-tts (hi-IN-SwaraNeural voice)
  |-- LLM: plain requests, ordered fallback chain
  |     ollama (local) -> groq -> gemini, configured via env
  v
session store (SQLite, data/ispar.db) -> per-question evaluations + final report
```

Audio path: browser MediaRecorder (webm) -> POST /api/session/{sid}/answer ->
whisper.cpp transcribes -> LLM evaluates -> JSON feedback.
Question audio: GET /api/session/{sid}/question/{qid}/audio returns MP3 from edge-tts.

## Setup (Windows 11 laptop)

1. Install Python 3.12 and make sure `python` is on PATH.

2. Create and activate a virtual env, then install deps:
   ```
   cd interview-sparring-partner
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Start the whisper.cpp server (speech-to-text) on port 8080 with a ggml model:
   ```
   whisper-server.exe -m path\to\ggml-model.bin --port 8080
   ```
   Keep this terminal open while using the app.

4. LLM — default is **Groq cloud** (`openai/gpt-oss-120b`):
   - Copy `.env.example` to `.env` and put your key in `GROQ_API_KEY`.
     `GEMINI_API_KEY` is optional — Gemini (`gemini-3.6-flash`) is the automatic
     fallback if Groq rate-limits (chain order: `ISP_LLM_CHAIN=groq,gemini`).
   - Prefer local instead? Set `ISP_LLM_PROVIDER=ollama`, install Ollama, then
     `ollama pull cyberforge-v2` (or your own model) and keep Ollama running
     on `http://localhost:11434`.

5. Start the app from the repo root so the `ispar` package is importable:
   ```
   set PYTHONPATH=src        :: cmd
   $env:PYTHONPATH="src"     :: PowerShell
   uvicorn ispar.app:app --port 8000
   ```

6. Open http://127.0.0.1:8000 in your browser, upload a resume PDF, allow
   microphone permission, and start sparring.

## API docs

| Method | Endpoint | Request | Response |
|---|---|---|---|
| POST | `/api/resume/upload` | multipart form: `file` (PDF) | `{profile: {name, email, phone, skills[], projects[{title, tech, bullets[]}], ...}, questions: [{id, category, difficulty, question, ideal_points[]}]}` |
| POST | `/api/session/start` | JSON: `{profile, questions}` | `{session_id}` |
| GET | `/api/session/{sid}/question/{qid}/audio` | — | `audio/mpeg` (question spoken in Hinglish TTS) |
| POST | `/api/session/{sid}/answer` | multipart: `qid`, `audio` (webm) | `{transcript, evaluation: {score_1_10, content_feedback, structure_feedback, clarity_feedback, filler_count, improvements[3], model_answer_2lines}}` |
| GET | `/api/session/{sid}/report` | — | `{per_question[{qid, question, score, improvements}], avg_score, total_fillers, top_improvements[3], weakest_model_answers[{qid, question, model_answer_2lines}]}` |

## Env vars

| Variable | Default | Purpose |
|---|---|---|
| `ISP_LLM_PROVIDER` | `ollama` | Primary LLM provider: `ollama`, `groq`, or `gemini` |
| `ISP_OLLAMA_URL` | `http://localhost:11434` | Local Ollama base URL |
| `ISP_OLLAMA_MODEL` | `cyberforge-v2` | Ollama model name for question gen + evaluation |
| `ISP_LLM_CHAIN` | `groq,gemini` | Ordered fallback providers if the primary fails |
| `GROQ_API_KEY` | — | Groq cloud API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model name |
| `GEMINI_API_KEY` | — | Google Gemini cloud API key |
| `WHISPER_URL` | `http://127.0.0.1:8080` | whisper.cpp server URL (speech-to-text) |
| `ISP_TRANSCRIBE` | `local` | Speech-to-text backend: `local` (whisper.cpp) or `groq` (hosted Whisper API) |
| `GROQ_WHISPER_MODEL` | `whisper-large-v3-turbo` | Groq Whisper model (groq transcription backend only) |
| `TTS_VOICE` | `hi-IN-SwaraNeural` | edge-tts voice for spoken questions |

## Troubleshooting

- **Answer submission returns 502 / "whisper not running"**: the whisper.cpp server
  isn't up. Start `whisper-server.exe ... --port 8080` and make sure `WHISPER_URL`
  points at it.
- **Question generation fails / 502 from the LLM**: no LLM is reachable. Check that
  Ollama is running with the model pulled (`ollama list`), or that the cloud keys
  in `.env` are set and `ISP_LLM_CHAIN` names a provider with a key.
- **Microphone not recording in the browser**: the site needs mic permission.
  Click the lock/tune icon in the address bar and allow the microphone, then reload.
  Note: browsers only allow mic access on `localhost` or HTTPS.
- **`ModuleNotFoundError: ispar`**: you started uvicorn from the wrong folder or
  without `PYTHONPATH=src`. Start from the repo root with the env var set as shown
  in step 5 above.
