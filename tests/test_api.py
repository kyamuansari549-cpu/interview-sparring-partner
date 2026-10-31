import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ispar.app as app_module
from ispar.llm import LLMError
from ispar.session_store import SessionStore

FAKE_PROFILE = {"name": "Test Candidate", "skills": ["Python", "FastAPI"]}
FAKE_QUESTIONS = [
    {
        "id": "q1",
        "category": "technical",
        "difficulty": "medium",
        "question": "Explain how FastAPI works.",
        "ideal_points": ["async", "pydantic", "openapi"],
    },
    {
        "id": "q2",
        "category": "behavioral",
        "difficulty": "easy",
        "question": "Tell me about yourself.",
        "ideal_points": ["background", "strengths"],
    },
]
FAKE_EVALUATION = {
    "score_1_10": 7,
    "content_feedback": "good",
    "structure_feedback": "ok",
    "clarity_feedback": "clear",
    "improvements": ["Slow down", "Add examples", "Slow down"],
    "model_answer_2lines": "FastAPI is async. It uses Pydantic.",
    "filler_count": 3,
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    test_store = SessionStore(db_path=str(tmp_path / "test.db"))
    monkeypatch.setattr(app_module, "store", test_store)
    monkeypatch.setattr(
        app_module, "parse_resume", lambda path: dict(FAKE_PROFILE)
    )
    monkeypatch.setattr(
        app_module,
        "generate_question_bank",
        lambda profile: [dict(q) for q in FAKE_QUESTIONS],
    )

    def fake_transcribe(path):
        return "fake transcript here"

    def fake_synthesize(text, mp3_path):
        mp3_path = Path(mp3_path)
        mp3_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_path.write_bytes(b"ID3fake-mp3")
        return mp3_path

    monkeypatch.setattr(app_module, "transcribe_whisper", fake_transcribe)
    monkeypatch.setattr(app_module, "synthesize_edge", fake_synthesize)
    monkeypatch.setattr(
        app_module, "evaluate_answer", lambda q, pts, t: dict(FAKE_EVALUATION)
    )
    monkeypatch.setattr(app_module, "AUDIO_DIR", tmp_path / "audio")

    with TestClient(app_module.app) as c:
        yield c
    test_store.close()


def _start_session(client):
    r = client.post(
        "/api/session/start",
        json={"profile": FAKE_PROFILE, "questions": FAKE_QUESTIONS},
    )
    assert r.status_code == 200
    return r.json()["session_id"]


# ---------------------------------------------------------------- upload ---


def test_upload_rejects_non_pdf(client):
    r = client.post(
        "/api/resume/upload",
        files={"file": ("resume.txt", b"not a pdf", "text/plain")},
    )
    assert r.status_code == 400


def test_upload_happy_path(client):
    r = client.post(
        "/api/resume/upload",
        files={"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["profile"] == FAKE_PROFILE
    assert body["questions"] == FAKE_QUESTIONS


def test_upload_llm_down_returns_502(client, monkeypatch):
    def boom(profile):
        raise LLMError("groq failed: boom")

    monkeypatch.setattr(app_module, "generate_question_bank", boom)
    r = client.post(
        "/api/resume/upload",
        files={"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert r.status_code == 502
    assert "groq" in r.json()["detail"]


# ---------------------------------------------------------------- start ---


def test_session_start_happy_path(client):
    r = client.post(
        "/api/session/start",
        json={"profile": FAKE_PROFILE, "questions": FAKE_QUESTIONS},
    )
    assert r.status_code == 200
    assert r.json()["session_id"]


def test_session_start_empty_questions_400(client):
    r = client.post(
        "/api/session/start",
        json={"profile": FAKE_PROFILE, "questions": []},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------- audio ---


def test_question_audio_unknown_sid_404(client):
    r = client.get("/api/session/nonexistent/question/q1/audio")
    assert r.status_code == 404


def test_question_audio_unknown_qid_404(client):
    sid = _start_session(client)
    r = client.get(f"/api/session/{sid}/question/nope/audio")
    assert r.status_code == 404


def test_question_audio_happy_path(client):
    sid = _start_session(client)
    r = client.get(f"/api/session/{sid}/question/q1/audio")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert r.content.startswith(b"ID3")


# ---------------------------------------------------------------- answer ---


def test_answer_unknown_sid_404(client):
    r = client.post(
        "/api/session/nonexistent/answer",
        data={"qid": "q1"},
        files={"audio": ("a.webm", b"bytes", "audio/webm")},
    )
    assert r.status_code == 404


def test_answer_empty_audio_400(client):
    sid = _start_session(client)
    r = client.post(
        f"/api/session/{sid}/answer",
        data={"qid": "q1"},
        files={"audio": ("a.webm", b"", "audio/webm")},
    )
    assert r.status_code == 400


def test_answer_happy_path(client):
    sid = _start_session(client)
    r = client.post(
        f"/api/session/{sid}/answer",
        data={"qid": "q1"},
        files={"audio": ("a.webm", b"audiobytes", "audio/webm")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "fake transcript here"
    assert body["evaluation"] == FAKE_EVALUATION


# ---------------------------------------------------------------- report ---


def test_report_after_one_answer(client):
    sid = _start_session(client)
    client.post(
        f"/api/session/{sid}/answer",
        data={"qid": "q1"},
        files={"audio": ("a.webm", b"audiobytes", "audio/webm")},
    )
    r = client.get(f"/api/session/{sid}/report")
    assert r.status_code == 200
    body = r.json()
    assert body["avg_score"] == 7.0
    assert body["total_fillers"] == 3
    assert len(body["per_question"]) == 1
    pq = body["per_question"][0]
    assert pq["qid"] == "q1"
    assert pq["score"] == 7
    # top_improvements must be deduped & frequency-ranked
    assert body["top_improvements"][0] == "Slow down"
    assert len(body["top_improvements"]) == 2
    assert len(body["weakest_model_answers"]) == 1
    assert body["weakest_model_answers"][0]["qid"] == "q1"


def test_report_unknown_sid_404(client):
    r = client.get("/api/session/nonexistent/report")
    assert r.status_code == 404
