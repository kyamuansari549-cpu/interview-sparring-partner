"""The FastAPI app — five endpoints plus the static UI.

  POST /api/resume/upload                      resume PDF -> profile + question bank
  POST /api/session/start                      -> session_id
  GET  /api/session/{sid}/question/{qid}/audio question as MP3 (edge-tts)
  POST /api/session/{sid}/answer               voice answer -> transcript + evaluation
  GET  /api/session/{sid}/report               the final scorecard data

The UI mount at "/" comes last on purpose: mount it earlier and
StaticFiles happily shadows the /api routes.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import DATA_DIR
from .evaluator import EvaluationError, evaluate_answer
from .llm import LLMError
from .question_bank import QuestionBankError, generate_question_bank
from .resume_parser import parse_resume
from .session_store import SessionStore
from .voice import VoiceError, synthesize_edge, transcribe_whisper

app = FastAPI(title="Interview Sparring Partner")

store = SessionStore()
AUDIO_DIR = Path(DATA_DIR) / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- resume ---


@app.post("/api/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf resumes accepted")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    with tempfile.NamedTemporaryFile(
        suffix=".pdf", prefix="ispar-resume-", delete=False
    ) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        try:
            profile = parse_resume(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            questions = generate_question_bank(profile)
        except (QuestionBankError, LLMError) as exc:
            raise HTTPException(
                status_code=502, detail=f"LLM question generation failed: {exc}"
            ) from exc
        return {"profile": profile, "questions": questions}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# --------------------------------------------------------------- session ---


@app.post("/api/session/start")
def start_session(body: dict):
    profile = body.get("profile") if isinstance(body, dict) else None
    questions = body.get("questions") if isinstance(body, dict) else None
    if not profile or not questions:
        raise HTTPException(
            status_code=400,
            detail="Both 'profile' and non-empty 'questions' are required",
        )
    sid = store.create_session(profile, questions)
    return {"session_id": sid}


def _session_or_404(sid: str) -> dict:
    session = store.get_session(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _question_or_404(session: dict, qid: str) -> dict:
    for q in session["questions"]:
        if q.get("id") == qid:
            return q
    raise HTTPException(status_code=404, detail="Question not found in session")


# ----------------------------------------------------------------- audio ---


@app.get("/api/session/{sid}/question/{qid}/audio")
def question_audio(sid: str, qid: str):
    session = _session_or_404(sid)
    question = _question_or_404(session, qid)
    mp3_path = AUDIO_DIR / f"{sid}_{qid}.mp3"
    if not mp3_path.exists():
        try:
            synthesize_edge(question.get("question", ""), mp3_path)
        except VoiceError as exc:
            raise HTTPException(
                status_code=502, detail=f"edge-tts failed: {exc}"
            ) from exc
    return FileResponse(mp3_path, media_type="audio/mpeg", filename=mp3_path.name)


# ---------------------------------------------------------------- answer ---


@app.post("/api/session/{sid}/answer")
async def submit_answer(sid: str, qid: str = Form(...), audio: UploadFile = File(...)):
    session = _session_or_404(sid)
    question = _question_or_404(session, qid)

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    suffix = Path(audio.filename or "answer.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(
        suffix=suffix, prefix="ispar-answer-", delete=False
    ) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        try:
            transcript = transcribe_whisper(tmp_path)
        except VoiceError as exc:
            raise HTTPException(
                status_code=502, detail=f"whisper.cpp failed: {exc}"
            ) from exc
        if not transcript or not transcript.strip():
            raise HTTPException(status_code=400, detail="no speech detected")

        ideal_points = question.get("ideal_points") or []
        try:
            evaluation = evaluate_answer(
                question.get("question", ""), ideal_points, transcript.strip()
            )
        except (EvaluationError, LLMError) as exc:
            raise HTTPException(
                status_code=502, detail=f"LLM evaluation failed: {exc}"
            ) from exc

        store.save_answer(sid, qid, transcript.strip(), evaluation)
        return {"transcript": transcript.strip(), "evaluation": evaluation}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------- report ---


@app.get("/api/session/{sid}/report")
def session_report(sid: str):
    session = _session_or_404(sid)
    questions = {q.get("id"): q for q in session["questions"]}
    answers = store.get_answers(sid)

    per_question: list[dict] = []
    filler_total = 0
    improvement_counter: Counter = Counter()
    scores: list[tuple[str, int]] = []
    model_answers: dict[str, str] = {}

    for ans in answers:
        qid = ans["qid"]
        question = questions.get(qid)
        if question is None:
            continue  # skip answers whose question is unknown
        ev = ans["evaluation"] or {}
        score = ev.get("score_1_10", 0)
        improvements = ev.get("improvements") or []
        filler_total += ev.get("filler_count", 0)
        improvement_counter.update([i for i in improvements if isinstance(i, str)])
        per_question.append(
            {
                "qid": qid,
                "question": question.get("question", ""),
                "score": score,
                "improvements": improvements,
            }
        )
        scores.append((qid, score))
        model_answers[qid] = ev.get("model_answer_2lines", "")

    avg_score = round(sum(s for _, s in scores) / len(scores), 1) if scores else 0.0
    top_improvements = [
        imp for imp, _count in improvement_counter.most_common(3)
    ]
    weakest = sorted(scores, key=lambda pair: pair[1])[:2]
    weakest_model_answers = [
        {
            "qid": qid,
            "question": questions[qid].get("question", ""),
            "model_answer_2lines": model_answers.get(qid, ""),
        }
        for qid, _score in weakest
    ]

    return {
        "session_id": sid,
        "answered": len(per_question),
        "total_questions": len(session["questions"]),
        "avg_score": avg_score,
        "total_fillers": filler_total,
        "per_question": per_question,
        "top_improvements": top_improvements,
        "weakest_model_answers": weakest_model_answers,
    }


# ---------------------------------------------------------------- static ---

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ispar.app:app", host="127.0.0.1", port=8000)
