"""Voice I/O for the Interview Sparring Partner.

transcribe_whisper(audio_path) -> str   (speech-to-text, default Hindi)
synthesize_edge(text, mp3_path) -> Path (edge-tts, mp3 output for the web UI)

Two transcription backends, picked by ISP_TRANSCRIBE in config: the local
whisper.cpp server (the laptop setup), or Groq's hosted Whisper API (the
Render deploy, where no local server can run). Web clients record webm/opus,
so on the local path non-WAV input is converted to 16kHz mono WAV with the
bundled imageio-ffmpeg binary before being sent to whisper.cpp; the Groq
API accepts the recording as-is.
"""

import subprocess
import tempfile
from pathlib import Path

import requests

import imageio_ffmpeg


class VoiceError(Exception):
    """Raised when a voice dependency fails. Message names the dependency."""


def _to_wav_16k(src: str | Path) -> Path:
    """Convert any audio file to 16kHz mono WAV in a temp file."""
    src = Path(src)
    fd, tmp = tempfile.mkstemp(prefix="ispar-", suffix=".wav")
    import os

    os.close(fd)
    tmp_path = Path(tmp)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        raise VoiceError(
            f"whisper.cpp input conversion failed (ffmpeg could not convert "
            f"{src.name} to WAV): {exc.stderr.strip() or exc}"
        ) from exc
    except FileNotFoundError as exc:
        tmp_path.unlink(missing_ok=True)
        raise VoiceError(
            f"whisper.cpp input conversion failed (ffmpeg binary missing): {exc}"
        ) from exc
    return tmp_path


def transcribe_whisper(
    audio_path: str | Path,
    server_url: str | None = None,
    language: str = "hi",
) -> str:
    """Transcribe an audio file (backend chosen by ISP_TRANSCRIBE)."""
    # Note to self: on the local path, whisper is a separate process that
    # has to be started by hand (README, setup step 3). When local
    # transcription starts failing, it is almost never the code — the
    # server just isn't running.
    from . import config

    if config.TRANSCRIBE_PROVIDER == "groq":
        return _transcribe_groq(audio_path, language)

    audio_path = Path(audio_path)
    if server_url is None:
        server_url = config.WHISPER_URL

    temp_wav: Path | None = None
    try:
        if audio_path.suffix.lower() == ".wav":
            wav_path = audio_path
        else:
            temp_wav = _to_wav_16k(audio_path)
            wav_path = temp_wav

        with open(wav_path, "rb") as f:
            try:
                r = requests.post(
                    server_url.rstrip("/") + "/inference",
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={"language": language, "temperature": "0.0"},
                    timeout=120,
                )
            except requests.RequestException as exc:
                raise VoiceError(
                    f"whisper.cpp transcription failed (network error talking to "
                    f"{server_url}): {exc}"
                ) from exc

        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            raise VoiceError(
                f"whisper.cpp transcription failed (HTTP {r.status_code} from "
                f"{server_url}): {r.text[:200]}"
            ) from exc

        try:
            return r.json()["text"].strip()
        except (ValueError, KeyError, TypeError) as exc:
            raise VoiceError(
                f"whisper.cpp transcription failed (bad JSON response from "
                f"{server_url}): {exc}"
            ) from exc
    finally:
        if temp_wav is not None:
            temp_wav.unlink(missing_ok=True)


def _transcribe_groq(audio_path: str | Path, language: str) -> str:
    """Transcribe with Groq's hosted Whisper API (OpenAI-compatible).

    Used when ISP_TRANSCRIBE=groq — hosted deploys with no local
    whisper.cpp. The recording is sent as-is (webm/mp3/wav are all
    accepted), so this path needs no ffmpeg conversion.
    """
    from . import config

    keys = config._provider_keys("groq")
    if not keys:
        raise VoiceError(
            "Groq transcription failed (ISP_TRANSCRIBE=groq but "
            "GROQ_API_KEY is not set)"
        )
    audio_path = Path(audio_path)
    try:
        with open(audio_path, "rb") as f:
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {keys[0]}"},
                files={"file": (audio_path.name, f)},
                data={
                    "model": config.GROQ_WHISPER_MODEL,
                    "language": language,
                    "response_format": "json",
                    "temperature": "0.0",
                },
                timeout=120,
            )
    except requests.RequestException as exc:
        raise VoiceError(
            f"Groq transcription failed (network error): {exc}"
        ) from exc
    try:
        r.raise_for_status()
    except requests.HTTPError as exc:
        raise VoiceError(
            f"Groq transcription failed (HTTP {r.status_code}): "
            f"{r.text[:200]}"
        ) from exc
    try:
        return r.json()["text"].strip()
    except (ValueError, KeyError, TypeError) as exc:
        raise VoiceError(
            f"Groq transcription failed (bad JSON response): {exc}"
        ) from exc


def synthesize_edge(
    text: str, mp3_path: str | Path, voice: str | None = None
) -> Path:
    """Speak `text` into an MP3 file with edge-tts (no WAV conversion)."""
    from . import config

    import asyncio

    # Lazy import bound to the module attribute so tests can monkeypatch
    # ispar.voice.edge_tts without paying the import cost at module load.
    global edge_tts
    if "edge_tts" not in globals():
        import edge_tts  # noqa: F401

    mp3_path = Path(mp3_path)
    if voice is None:
        voice = config.TTS_VOICE
    mp3_path.parent.mkdir(parents=True, exist_ok=True)

    async def _save() -> None:
        await edge_tts.Communicate(text, voice).save(str(mp3_path))

    try:
        asyncio.run(_save())
    except Exception as exc:
        raise VoiceError(f"edge-tts synthesis failed: {exc}") from exc

    return mp3_path
