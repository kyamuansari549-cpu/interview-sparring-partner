import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import requests  # noqa: E402

from ispar import voice  # noqa: E402
from ispar.voice import VoiceError, synthesize_edge, transcribe_whisper  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = str(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _write_wav(tmp_path) -> Path:
    wav = tmp_path / "in.wav"
    # Minimal RIFF header is enough; requests.post is mocked, file is never read.
    wav.write_bytes(b"RIFF" + b"\x00" * 40)
    return wav


def test_transcribe_ok(monkeypatch, tmp_path):
    wav = _write_wav(tmp_path)
    calls = {}

    def fake_post(url, files=None, data=None, timeout=None):
        calls["url"] = url
        calls["files"] = files
        calls["data"] = data
        calls["timeout"] = timeout
        return FakeResponse({"text": "  namaste  "})

    monkeypatch.setattr(voice.requests, "post", fake_post)

    assert transcribe_whisper(wav) == "namaste"
    assert calls["url"].endswith("/inference")
    assert set(calls["files"].keys()) == {"file"}
    assert calls["files"]["file"][0] == "audio.wav"
    assert calls["files"]["file"][2] == "audio/wav"
    assert calls["data"]["language"] == "hi"
    assert "temperature" in calls["data"]


def test_transcribe_network_failure(monkeypatch, tmp_path):
    wav = _write_wav(tmp_path)

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(voice.requests, "post", fake_post)

    with pytest.raises(VoiceError) as exc_info:
        transcribe_whisper(wav)
    assert "whisper" in str(exc_info.value).lower()


def test_transcribe_webm_converts_to_wav(monkeypatch, tmp_path):
    webm = tmp_path / "in.webm"
    webm.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 40)
    converted = tmp_path / "converted.wav"
    converted.write_bytes(b"RIFF" + b"\x00" * 40)

    def fake_to_wav(src):
        assert Path(src).suffix == ".webm"
        return converted

    monkeypatch.setattr(voice, "_to_wav_16k", fake_to_wav)

    posted = {}

    def fake_post(url, files=None, data=None, timeout=None):
        posted["file_obj"] = files["file"][1]
        return FakeResponse({"text": "hello"})

    monkeypatch.setattr(voice.requests, "post", fake_post)

    assert transcribe_whisper(webm) == "hello"
    # The file object posted must be the converted WAV, not the original webm.
    assert posted["file_obj"].name == str(converted)


class FakeEdgeTTS:
    """Fake edge_tts module: Communicate(...).save(path) writes bytes."""

    saved_voice = None

    class Communicate:
        def __init__(self, text, voice):
            self.text = text
            self.voice = voice

        async def save(self, path):
            FakeEdgeTTS.saved_voice = self.voice
            Path(path).write_bytes(b"ID3\x04\x00\x00\x00fake-mp3-bytes")


def test_synthesize_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(voice, "edge_tts", FakeEdgeTTS, raising=False)
    out = tmp_path / "nested" / "out.mp3"

    result = synthesize_edge("namaste", out, voice="hi-IN-SwaraNeural")

    assert result == Path(out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes().startswith(b"ID3")
    assert FakeEdgeTTS.saved_voice == "hi-IN-SwaraNeural"


def test_synthesize_uses_config_default_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(voice, "edge_tts", FakeEdgeTTS, raising=False)
    out = tmp_path / "out.mp3"

    synthesize_edge("namaste", out)

    from ispar import config

    assert FakeEdgeTTS.saved_voice == config.TTS_VOICE


class FailingEdgeTTS:
    class Communicate:
        def __init__(self, text, voice):
            pass

        async def save(self, path):
            raise RuntimeError("no network")


def test_synthesize_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(voice, "edge_tts", FailingEdgeTTS, raising=False)

    with pytest.raises(VoiceError) as exc_info:
        synthesize_edge("namaste", tmp_path / "out.mp3")
    assert "edge-tts" in str(exc_info.value).lower()
