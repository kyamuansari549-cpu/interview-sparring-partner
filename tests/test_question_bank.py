import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json  # noqa: E402

import pytest  # noqa: E402

from ispar import interviewer  # noqa: E402
from ispar import question_bank  # noqa: E402
from ispar.question_bank import CATEGORIES, QuestionBankError, generate_question_bank  # noqa: E402

PROFILE = {
    "name": "Kiyamuddin Ansari",
    "skills": ["Python", "Java", "Next.js", "PyTorch"],
    "projects": [
        {
            "name": "ObjectDetectApp",
            "tech": ["Python", "PyTorch", "OpenCV"],
            "bullets": [
                "Fine-tuned EfficientDet-Lite0 on a custom dataset",
                "Used NMS with IoU threshold 0.5 for post-processing",
            ],
        },
        {
            "name": "SeatBook",
            "tech": ["Next.js", "Postgres", "Redis"],
            "bullets": [
                "Seat holds implemented with Redis TTL",
                "Razorpay payment integration with webhook confirmation",
            ],
        },
    ],
}


def _make_question(qid, category, difficulty="medium"):
    return {
        "id": qid,
        "category": category,
        "difficulty": difficulty,
        "question": f"Sample interview question {qid}?",
        "ideal_points": ["first key point", "second key point", "third key point"],
    }


def _valid_bank():
    cats = (
        ["resume_deep_dive"] * 4
        + ["cs_fundamentals"] * 3
        + ["hr"] * 2
        + ["wildcard"]
    )
    return [_make_question(f"q{i + 1}", cat) for i, cat in enumerate(cats)]


def test_happy_path(monkeypatch):
    bank_json = json.dumps(_valid_bank())
    calls = []

    def fake_chat(messages, temperature=0.7, json_mode=False):
        calls.append({"json_mode": json_mode, "temperature": temperature})
        return bank_json

    monkeypatch.setattr(question_bank, "chat", fake_chat)

    bank = generate_question_bank(PROFILE)

    assert len(bank) == 10
    assert [q["id"] for q in bank] == [f"q{i + 1}" for i in range(10)]
    counts = {}
    for q in bank:
        assert q["category"] in CATEGORIES
        counts[q["category"]] = counts.get(q["category"], 0) + 1
        assert 2 <= len(q["ideal_points"]) <= 4
        assert all(isinstance(p, str) and p.strip() for p in q["ideal_points"])
    assert counts == {
        "resume_deep_dive": 4,
        "cs_fundamentals": 3,
        "hr": 2,
        "wildcard": 1,
    }
    assert calls[0]["json_mode"] is True


def test_repair_path(monkeypatch):
    calls = []

    def fake_chat(messages, temperature=0.7, json_mode=False):
        calls.append(messages)
        if len(calls) == 1:
            return "this is not JSON at all {{{"
        return json.dumps(_valid_bank())

    monkeypatch.setattr(question_bank, "chat", fake_chat)

    bank = generate_question_bank(PROFILE)

    assert len(bank) == 10
    assert len(calls) == 2
    # The repair prompt must reference the invalid output.
    repair_user_text = calls[1][-1]["content"]
    assert "invalid" in repair_user_text.lower()
    assert "ONLY" in repair_user_text


def test_failure_path(monkeypatch):
    calls = []

    def fake_chat(messages, temperature=0.7, json_mode=False):
        calls.append(1)
        return "still not JSON {{{"

    monkeypatch.setattr(question_bank, "chat", fake_chat)

    with pytest.raises(QuestionBankError):
        generate_question_bank(PROFILE)
    assert len(calls) == 2


def test_schema_violation_missing_ideal_points(monkeypatch):
    bank = _valid_bank()
    for q in bank:
        del q["ideal_points"]

    def fake_chat(messages, temperature=0.7, json_mode=False):
        return json.dumps(bank)

    monkeypatch.setattr(question_bank, "chat", fake_chat)

    with pytest.raises(QuestionBankError) as excinfo:
        generate_question_bank(PROFILE)
    assert "ideal_points" in str(excinfo.value)


def test_build_followup(monkeypatch):
    expected = "Achha, toh isme NMS ka threshold tune kaise kiya tha?"

    def fake_chat(messages, temperature=0.7, json_mode=False):
        return "  " + expected + "  "

    monkeypatch.setattr(interviewer, "chat", fake_chat)

    result = interviewer.build_followup(
        "Tell me about your object detection project.",
        "maine EfficientDet use kiya tha aur NMS lagaya tha",
    )
    assert result == expected


def test_system_prompt_persona():
    assert "Hinglish" in interviewer.SYSTEM_PROMPT
    assert "40" in interviewer.SYSTEM_PROMPT
    assert "one question at a time" in interviewer.SYSTEM_PROMPT
