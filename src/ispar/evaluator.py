"""Answer evaluator for the Interview Sparring Partner MVP.

Scores a candidate's transcribed spoken answer against the interview
question and a list of ideal answer points, using an LLM guided by a
fixed rubric. Filler-word counting is always done deterministically in
Python — the LLM is never trusted for counts, and its reply is validated
before use.
"""

from __future__ import annotations

import json
import re

from .llm import LLMError, chat

FILLER_WORDS = [
    "umm",
    "uh",
    "like",
    "matlab",
    "toh",
    "actually",
    "basically",
    "you know",
    "hmm",
]

# Longest-first alternation so the multi-word phrase "you know" wins over
# any shorter match; \b on both ends keeps e.g. "toh" from matching inside
# a longer word while still matching across punctuation ("like-like" -> 2).
_FILLER_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(w) for w in sorted(FILLER_WORDS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def count_fillers(transcript: str) -> int:
    """Count filler words/phrases in *transcript*.

    Case-insensitive and word-boundary aware; the multi-word phrase
    "you know" counts once per occurrence. Non-overlapping by construction.
    """
    if not transcript:
        return 0
    return len(_FILLER_RE.findall(transcript))


class EvaluationError(Exception):
    """Raised when an answer cannot be evaluated."""


_EVAL_SCHEMA = (
    '{"score_1_10": <int 1-10>, "content_feedback": "<1-2 lines>", '
    '"structure_feedback": "<1-2 lines>", "clarity_feedback": "<1-2 lines>", '
    '"improvements": ["<short tip 1>", "<short tip 2>", "<short tip 3>"], '
    '"model_answer_2lines": "<two-line ideal answer>"}'
)


def _build_messages(
    question: str, ideal_points: list[str], transcript: str
) -> list[dict[str, str]]:
    points = "\n".join(f"- {p}" for p in ideal_points) or "- (none provided)"
    return [
        {
            "role": "system",
            "content": (
                "You are a strict but fair interview coach evaluating mock-interview "
                "answers. You always reply with strict JSON only — no markdown "
                "fences, no extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate the candidate's spoken interview answer (transcribed below).\n\n"
                f"Interview question:\n{question}\n\n"
                "Ideal answer points the candidate should cover:\n"
                f"{points}\n\n"
                "Candidate's transcript (verbatim — use ONLY what is written here):\n"
                f"{transcript}\n\n"
                "Rubric (weights sum to 100%):\n"
                "- Content accuracy — how well the answer covers the ideal points: 40%\n"
                "- Structure — logical flow / STAR method where relevant: 25%\n"
                "- Clarity — easy to follow, concise: 20%\n"
                "- Confidence — decisive language, minimal hedging: 15%\n\n"
                "Rules:\n"
                "- Base ALL feedback only on the transcript above.\n"
                "- Do NOT quote long passages from the transcript.\n"
                "- NEVER invent, assume, or add anything the candidate did not say.\n"
                "- Feedback strings may be in Hinglish.\n"
                "- Reply with STRICT JSON only, exactly this schema:\n"
                f"{_EVAL_SCHEMA}\n"
                '"improvements" must contain exactly 3 short strings.'
            ),
        },
    ]


_REPAIR_INSTRUCTION = (
    "Your previous reply was not valid. Reply again with STRICT JSON only — "
    "no markdown fences, no commentary — exactly this schema: "
    f"{_EVAL_SCHEMA} "
    "score_1_10 must be an integer from 1 to 10; improvements must be exactly "
    "3 short strings. Base everything only on the transcript already given; "
    "do not invent content."
)


def _coerce_score(value: object) -> int:
    """Coerce the LLM's score to an int and clamp it to 1..10."""
    if isinstance(value, bool):
        raise EvaluationError(f"invalid score_1_10: {value!r}")
    if isinstance(value, (int, float)):
        score = int(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            score = int(float(text)) if "." in text else int(text)
        except ValueError:
            raise EvaluationError(f"invalid score_1_10: {value!r}") from None
    else:
        raise EvaluationError(f"invalid score_1_10: {value!r}")
    return max(1, min(10, score))


def _parse_evaluation(raw: str) -> dict[str, object]:
    """Parse and validate one LLM reply; raises EvaluationError if invalid."""
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Tolerate surrounding prose: retry on the outermost {...} span.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise EvaluationError("LLM returned invalid JSON") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"LLM returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EvaluationError("LLM response is not a JSON object")

    score = _coerce_score(data.get("score_1_10"))

    def _feedback(key: str) -> str:
        value = data.get(key, "")
        return value if isinstance(value, str) else str(value)

    improvements = data.get("improvements")
    if not isinstance(improvements, list) or not all(
        isinstance(item, str) for item in improvements
    ):
        raise EvaluationError("LLM response has invalid 'improvements'")
    tips = list(improvements[:3])
    while len(tips) < 3:
        # Pad without inventing advice: empty slots rather than new tips.
        tips.append("")

    return {
        "score_1_10": score,
        "content_feedback": _feedback("content_feedback"),
        "structure_feedback": _feedback("structure_feedback"),
        "clarity_feedback": _feedback("clarity_feedback"),
        "improvements": tips,
        "model_answer_2lines": _feedback("model_answer_2lines"),
    }


def evaluate_answer(
    question: str, ideal_points: list[str], transcript: str
) -> dict[str, object]:
    """Evaluate a transcribed spoken answer.

    Raises EvaluationError for empty transcripts, LLM failures, or when the
    LLM cannot produce valid JSON even after one repair attempt. The returned
    ``filler_count`` is always computed in Python, overriding any LLM value.
    """
    if not transcript or not transcript.strip():
        raise EvaluationError("empty transcript — no speech detected")

    filler_count = count_fillers(transcript)  # never trust an LLM count
    messages = _build_messages(question, ideal_points, transcript)

    try:
        raw = chat(messages, temperature=0.3, json_mode=True)
    except LLMError as exc:
        raise EvaluationError(f"LLM call failed: {exc}") from exc

    try:
        result = _parse_evaluation(raw)
    except EvaluationError:
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _REPAIR_INSTRUCTION},
        ]
        try:
            raw = chat(repair_messages, temperature=0.3, json_mode=True)
        except LLMError as exc:
            raise EvaluationError(f"LLM repair call failed: {exc}") from exc
        result = _parse_evaluation(raw)  # raises EvaluationError if still invalid

    # Python-computed count wins over anything the LLM may have emitted.
    result["filler_count"] = filler_count
    return result
