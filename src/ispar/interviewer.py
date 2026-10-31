"""Interviewer persona for the Interview Sparring Partner (ispar).

The interviewer speaks in natural Hinglish like a real Indian IT interviewer —
friendly, but sharp enough to probe vague answers with follow-ups.
"""

from __future__ import annotations

from .llm import chat

SYSTEM_PROMPT: str = (
    "You are a friendly but sharp technical interviewer at an Indian IT company. "
    "Speak naturally in Hinglish (Hindi+English mix) like real Indian interviews — "
    "never robotic English, never pure Hindi. Ask one question at a time, "
    "ask sharp follow-ups when answers are vague, keep each spoken question under 40 words."
)


def build_followup(question: str, transcript: str) -> str:
    """Generate one sharp Hinglish follow-up probing the weakest part of an answer.

    Args:
        question: the interview question that was asked.
        transcript: the candidate's spoken answer (STT text).

    Returns:
        The follow-up question text (at most ~40 words, Hinglish).

    Raises:
        LLMError: if the underlying LLM call fails.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You just asked this interview question:\n"
                f"{question}\n\n"
                "The candidate answered (transcript):\n"
                f"{transcript}\n\n"
                "Produce ONE sharp follow-up question that probes the weakest or vaguest "
                "part of their answer. Rules: Hinglish only, a single question, "
                "no feedback or commentary, no preamble — at most 40 words."
            ),
        },
    ]
    return chat(messages, temperature=0.7).strip()
