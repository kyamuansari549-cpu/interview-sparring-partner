import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from ispar.resume_parser import parse_resume

RESUME_PDF = os.path.expanduser("~/workspace/your_files/Kiyamuddin_Ansari_Resume_v2.pdf")


@pytest.fixture(scope="module")
def parsed():
    return parse_resume(RESUME_PDF)


def test_contact_fields(parsed):
    assert parsed["name"] == "KIYAMUDDIN ANSARI"
    assert parsed["email"] == "kyamuansari549@gmail.com"
    assert "8081315140" in parsed["phone"]


def test_skills(parsed):
    assert "Python" in parsed["skills"]
    assert len(parsed["skills"]) >= 10


def test_projects(parsed):
    projects = parsed["projects"]
    assert len(projects) == 4
    titles = " | ".join(p["title"] for p in projects)
    for expected in ("SeatBook", "DrishtiNav", "CyberForge", "Agentic Research Assistant"):
        assert expected in titles, f"missing project title: {expected}"
    for p in projects:
        assert len(p["bullets"]) >= 1, f"project has no bullets: {p['title']}"
    seatbook = next(p for p in projects if "SeatBook" in p["title"])
    assert "Next.js" in seatbook["tech"]


def test_experience(parsed):
    assert len(parsed["experience"]) >= 1
    assert "Academy of Skill Development" in parsed["experience"][0]["org"]


def test_education(parsed):
    assert len(parsed["education"]) >= 1
    text = " ".join(parsed["education"])
    assert "AKTU" in text or "School of Management Sciences" in text


def test_missing_file_raises_valueerror():
    with pytest.raises(ValueError):
        parse_resume("/tmp/does-not-exist-resume-xyz.pdf")
