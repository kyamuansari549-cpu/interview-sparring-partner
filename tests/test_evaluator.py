import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json

import pytest

from ispar import evaluator
from ispar.evaluator import EvaluationError, count_fillers, evaluate_answer


class TestCountFillers:
    @pytest.mark.parametrize(
        "transcript, expected",
        [
            ("Umm, matlab maine kaha tha, you know, actually...", 4),
            ("hello world", 0),
            ("hmm Hmm hmm", 3),
            ("like-like", 2),
            ("", 0),
            ("baton me toh", 1),
            # extra boundary checks beyond the spec
            ("ttoh", 0),  # "toh" must not match inside a longer word
            ("duh", 0),  # "uh" must not match inside a longer word
            ("you know you know", 2),  # phrase counted per occurrence
            ("You Know", 1),  # phrase is case-insensitive
            ("hmmm", 0),  # "hmm" must not match inside "hmmm"
        ],
    )
    def test_count_fillers(self, transcript, expected):
        assert count_fillers(transcript) == expected


def _valid_payload(**overrides):
    payload = {
        "score_1_10": 8,
        "content_feedback": "content theek tha",
        "structure_feedback": "STAR thoda missing tha",
        "clarity_feedback": "clear tha",
        "improvements": ["tip1", "tip2", "tip3"],
        "model_answer_2lines": "line one. line two.",
        "filler_count": 999,  # wrong on purpose: must be overridden
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_evaluate_happy_path_overrides_llm_filler_count(monkeypatch):
    captured = {}

    def fake_chat(messages, temperature=0.7, json_mode=False):
        captured["temperature"] = temperature
        captured["json_mode"] = json_mode
        captured["messages"] = messages
        return _valid_payload()

    monkeypatch.setattr(evaluator, "chat", fake_chat)

    transcript = "umm actually the answer is binary search"
    result = evaluate_answer("Explain binary search?", ["log n", "sorted array"], transcript)

    assert captured["json_mode"] is True
    assert captured["temperature"] == 0.3
    # prompt must forbid inventing transcript content
    prompt_text = " ".join(m["content"] for m in captured["messages"])
    assert "NEVER invent" in prompt_text
    assert transcript in prompt_text

    # filler_count comes from Python, never from the LLM's 999
    assert result["filler_count"] == count_fillers(transcript) == 2
    assert 1 <= result["score_1_10"] <= 10
    assert len(result["improvements"]) == 3
    assert result["content_feedback"] == "content theek tha"


def test_evaluate_score_clamped_and_improvements_truncated(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "chat",
        lambda messages, temperature=0.7, json_mode=False: _valid_payload(
            score_1_10=42, improvements=["a", "b", "c", "d", "e"]
        ),
    )
    result = evaluate_answer("Q?", ["p"], "some answer")
    assert result["score_1_10"] == 10
    assert result["improvements"] == ["a", "b", "c"]  # LLM's own strings, truncated


def test_evaluate_repair_retry_succeeds(monkeypatch):
    calls = []
    responses = ["not json at all", _valid_payload(score_1_10=7)]

    def fake_chat(messages, temperature=0.7, json_mode=False):
        calls.append(messages)
        return responses[len(calls) - 1]

    monkeypatch.setattr(evaluator, "chat", fake_chat)
    result = evaluate_answer("Q?", ["p"], "umm spoken answer")

    assert len(calls) == 2  # initial attempt + one repair retry
    assert result["score_1_10"] == 7
    # repair request carries the bad reply and the strict-JSON instruction
    repair_messages = calls[1]
    assert repair_messages[-2] == {"role": "assistant", "content": "not json at all"}
    assert "STRICT JSON" in repair_messages[-1]["content"]


def test_evaluate_empty_transcript_raises_without_llm_call(monkeypatch):
    def fake_chat(*args, **kwargs):
        raise AssertionError("chat must not be called for an empty transcript")

    monkeypatch.setattr(evaluator, "chat", fake_chat)
    for bad in ("", "   ", "\n\t "):
        with pytest.raises(EvaluationError, match="empty transcript"):
            evaluate_answer("Q?", ["p"], bad)


def test_evaluate_invalid_json_twice_raises(monkeypatch):
    calls = []

    def fake_chat(messages, temperature=0.7, json_mode=False):
        calls.append(messages)
        return "definitely not {{{ json"

    monkeypatch.setattr(evaluator, "chat", fake_chat)
    with pytest.raises(EvaluationError):
        evaluate_answer("Q?", ["p"], "some spoken answer")
    assert len(calls) == 2  # exactly one repair retry, then give up


def test_evaluate_llm_error_wrapped(monkeypatch):
    def fake_chat(*args, **kwargs):
        raise evaluator.LLMError("provider exploded")

    monkeypatch.setattr(evaluator, "chat", fake_chat)
    with pytest.raises(EvaluationError, match="LLM call failed"):
        evaluate_answer("Q?", ["p"], "some spoken answer")
