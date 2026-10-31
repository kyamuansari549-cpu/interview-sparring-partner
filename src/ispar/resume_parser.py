"""Resume PDF parser for the Interview Sparring Partner MVP.

Strategy: pdfplumber text extraction + deterministic regex heuristics first.
The LLM (``ispar.llm.chat``) is used only as a last-resort fallback for a
field whose heuristic parse came up empty, and it never fails hard — on any
error the parser returns whatever the heuristics found.

Note: this PDF renders bullet glyphs as the literal text "(cid:127)", so a
line starting with "(cid:127)" is treated as a bullet point.
"""
from __future__ import annotations

import re
from pathlib import Path

BULLET_PREFIX = "(cid:127)"

_SECTION_ALIASES = {
    "SUMMARY": "summary",
    "EDUCATION": "education",
    "INTERNSHIP": "experience",
    "EXPERIENCE": "experience",
    "PROJECTS": "projects",
    "TECHNICAL SKILLS": "skills",
    "SKILLS": "skills",
    "CERTIFICATIONS": "certifications",
    "ACHIEVEMENTS": "achievements",
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?91[\s\-]?\d{10}")
_PHONE_FALLBACK_RE = re.compile(r"(?<![\d+])[6-9]\d{9}(?!\d)")
_MONTHS = (
    "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    "|January|February|March|April|June|July|August|September|October|November|December"
)
_DATE_RANGE_RE = re.compile(
    rf"\s*(?:{_MONTHS})[a-z]*\s+\d{{4}}\s*[–—-]\s*(?:(?:{_MONTHS})[a-z]*\s+\d{{4}}|Present|present)\s*$"
)
_LINK_TOKENS = {"github", "live demo", "demo", "website", "live"}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _read_pdf_text(pdf_path: str | Path) -> str:
    """Extract raw text from every page of the PDF."""
    path = Path(pdf_path)
    if not path.is_file():
        raise ValueError(f"Resume file not found: {path}")
    try:
        import pdfplumber
    except ImportError as exc:
        raise ValueError("pdfplumber is required to parse resumes") from exc
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except Exception as exc:
        raise ValueError(f"Could not read PDF {path}: {exc}") from exc
    if not text.strip():
        raise ValueError(f"PDF has no extractable text: {path}")
    return text


def _is_capsish(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.6


def _is_section_header(line: str) -> bool:
    return line.strip().rstrip(":").upper() in _SECTION_ALIASES


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    """Group non-empty lines under their section header."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        key = stripped.rstrip(":").upper()
        if key in _SECTION_ALIASES:
            current = _SECTION_ALIASES[key]
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(stripped)
    return sections


def _is_bullet(line: str) -> bool:
    return line.strip().startswith(BULLET_PREFIX)


def _clean_bullet(line: str) -> str:
    return line.strip()[len(BULLET_PREFIX):].strip(" \t-•")


def _is_link_line(line: str) -> bool:
    tokens = [t.strip().lower() for t in line.split("|")]
    return bool(tokens) and all(t in _LINK_TOKENS for t in tokens)


def _next_nonempty(lines: list[str], i: int) -> tuple[int, str]:
    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    return j, (lines[j].strip() if j < len(lines) else "")


def _strip_date_range(text: str) -> str:
    return _DATE_RANGE_RE.sub("", text).strip(" –—-")


# ---------------------------------------------------------------------------
# Field extractors (heuristics)
# ---------------------------------------------------------------------------

def _extract_name(lines: list[str]) -> str:
    for line in lines[:8]:
        stripped = line.strip()
        if not stripped or "@" in stripped or _is_section_header(stripped):
            continue
        if _is_capsish(stripped) and len(stripped.split()) >= 2:
            return " ".join(stripped.split())
    return ""


def _extract_email(text: str) -> str:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else ""


def _extract_phone(text: str) -> str:
    m = _PHONE_RE.search(text) or _PHONE_FALLBACK_RE.search(text)
    if not m:
        return ""
    digits = re.sub(r"\D", "", m.group(0))
    if len(digits) == 12 and digits.startswith("91"):
        return f"+91 {digits[2:]}"
    if len(digits) == 10:
        return f"+91 {digits}"
    return m.group(0).strip()


def _parse_skills(lines: list[str]) -> list[str]:
    """Skills from 'Languages: a, b' style category lines (with wrap lines)."""
    skills: list[str] = []
    seen: set[str] = set()
    buffer = ""
    categories: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            categories.append(buffer.strip())
        buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped or _is_bullet(stripped):
            continue
        if ":" in stripped and not stripped.lower().startswith("http"):
            flush()
            buffer = stripped
        else:
            # Wrapped continuation of the previous category line.
            buffer = f"{buffer} {stripped}" if buffer else stripped
    flush()

    for cat in categories:
        items = cat.split(":", 1)[1] if ":" in cat else cat
        for item in items.split(","):
            skill = item.strip()
            key = skill.lower()
            if skill and key not in seen:
                seen.add(key)
                skills.append(skill)
    return skills


def _parse_projects(lines: list[str]) -> list[dict]:
    """Title line, then a 'A | B | C' tech line, then bullets until the next."""
    projects: list[dict] = []
    current: dict | None = None

    def close() -> None:
        nonlocal current
        if current is not None:
            projects.append(current)
        current = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or _is_section_header(line):
            i += 1
            continue
        if _is_bullet(line):
            bullet = _clean_bullet(line)
            if current is None:
                current = {"title": "", "tech": "", "bullets": []}
            current["bullets"].append(bullet)
            i += 1
            continue
        j, nxt = _next_nonempty(lines, i)
        if "|" in nxt and not _is_link_line(nxt):
            # New project: title line followed by a tech line.
            close()
            current = {"title": line, "tech": nxt, "bullets": []}
            i = j + 1
            continue
        if _is_link_line(line):
            i += 1  # "GitHub | Live Demo" style line — not data.
            continue
        if current is not None and current["bullets"]:
            # Wrapped continuation of the previous bullet.
            current["bullets"][-1] = f"{current['bullets'][-1]} {line}"
        i += 1
    close()
    return [p for p in projects if p["title"] or p["bullets"]]


def _split_role_header(line: str) -> tuple[str, str]:
    """'Org — Title (dates)' -> (org, title without date range)."""
    for sep in ("—", "–", " - "):
        if sep in line:
            org, rest = line.split(sep, 1)
            return org.strip(), _strip_date_range(rest.strip())
    return "", _strip_date_range(line.strip())


def _parse_experience(lines: list[str]) -> list[dict]:
    entries: list[dict] = []
    current: dict | None = None

    def close() -> None:
        nonlocal current
        if current is not None:
            entries.append(current)
        current = None

    for line in lines:
        stripped = line.strip()
        if not stripped or _is_section_header(stripped):
            continue
        if _is_bullet(stripped):
            if current is None:
                current = {"title": "", "org": "", "bullets": []}
            current["bullets"].append(_clean_bullet(stripped))
        elif _is_link_line(stripped):
            continue
        else:
            org, title = _split_role_header(stripped)
            # A bare fragment (wrapped line) belongs to the previous bullet.
            looks_like_header = bool(org) or (
                current is not None and not current["bullets"]
            )
            if looks_like_header:
                close()
                current = {"title": title, "org": org, "bullets": []}
            elif current is not None and current["bullets"]:
                current["bullets"][-1] = f"{current['bullets'][-1]} {stripped}"
            elif current is not None:
                current["title"] = f"{current['title']} {stripped}".strip()
    close()
    return [e for e in entries if e["title"] or e["org"] or e["bullets"]]


def _parse_education(lines: list[str]) -> list[str]:
    entries: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or _is_bullet(stripped) or _is_section_header(stripped):
            continue
        # A line naming an institution (or the first line) starts a new entry;
        # fragments like "CGPA: 6.70" attach to the previous one.
        starts_new = (
            not entries
            or "(" in stripped
            or re.search(r"\b(19|20)\d{2}\b", stripped)
            or re.search(r"\b(B\.?Tech|M\.?Tech|Class|Bachelor|Master)\b", stripped, re.I)
        )
        if starts_new:
            entries.append(stripped)
        else:
            entries[-1] = f"{entries[-1]} {stripped}"
    return entries


# ---------------------------------------------------------------------------
# LLM fallback (last resort, never fails hard)
# ---------------------------------------------------------------------------

def _llm_json(prompt: str) -> dict | list | None:
    """Ask the local LLM for JSON; return None on any failure."""
    try:
        from .llm import chat  # lazy: only when a heuristic parse failed
    except Exception:
        return None
    try:
        raw = chat(prompt)
    except Exception:
        return None
    try:
        import json

        m = re.search(r"\{.*\}|\[.*\]", raw, re.S)
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def _fallback(field: str, hint: str, context: str) -> dict | list | str:
    """LLM fallback for one ambiguous field; heuristic value wins if present."""
    result = _llm_json(
        "You extract structured data from resume text. Reply with ONLY a JSON "
        f"value for the field '{field}' (hint: {hint}). No other text.\n\n{context}"
    )
    return result if result is not None else ({} if field.endswith("s") else "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_resume(pdf_path: str | Path) -> dict:
    """Parse a resume PDF into a structured dict.

    Never fails hard on one bad section — returns whatever was found.
    Raises ValueError if the file is missing or has no extractable text.
    """
    text = _read_pdf_text(pdf_path)
    lines = [ln for ln in text.splitlines()]
    sections = _split_sections(lines)
    header_text = "\n".join(lines[:12])

    result: dict = {
        "name": "",
        "email": _extract_email(text),
        "phone": _extract_phone(text),
        "skills": [],
        "projects": [],
        "experience": [],
        "education": [],
    }

    def safe(fn, *args, default):
        try:
            return fn(*args)
        except Exception:
            return default

    result["name"] = safe(_extract_name, lines, default="")
    result["skills"] = safe(_parse_skills, sections.get("skills", []), default=[])
    result["projects"] = safe(_parse_projects, sections.get("projects", []), default=[])
    result["experience"] = safe(
        _parse_experience, sections.get("experience", []), default=[]
    )
    result["education"] = safe(
        _parse_education, sections.get("education", []), default=[]
    )

    # LLM fallback only where the heuristics found nothing.
    if not result["name"]:
        fb = _fallback("name", "candidate full name", header_text)
        if isinstance(fb, str):
            result["name"] = fb
    if not result["skills"]:
        fb = _fallback("skills", "list of technical skill strings",
                       "\n".join(sections.get("skills", [])))
        if isinstance(fb, list):
            result["skills"] = [str(s) for s in fb]
    if not result["projects"]:
        fb = _fallback("projects", 'list of {"title","tech","bullets"}',
                       "\n".join(sections.get("projects", [])))
        if isinstance(fb, list):
            result["projects"] = fb
    if not result["experience"]:
        fb = _fallback("experience", 'list of {"title","org","bullets"}',
                       "\n".join(sections.get("experience", [])))
        if isinstance(fb, list):
            result["experience"] = fb
    if not result["education"]:
        fb = _fallback("education", "list of education line strings",
                       "\n".join(sections.get("education", [])))
        if isinstance(fb, list):
            result["education"] = [str(e) for e in fb]

    return result
