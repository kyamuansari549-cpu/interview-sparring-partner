"""Voice I/O for the Interview Sparring Partner.

transcribe_whisper(audio_path) -> str   (whisper.cpp server, default Hindi)
synthesize_edge(text, mp3_path) -> Path (edge-tts, mp3 output for the web UI)

Web clients record webm/opus, so non-WAV input is converted to 16kHz mono
WAV with the bundled imageio-ffmpeg binary before being sent to whisper.cpp.
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
    """Transcribe an audio file with the whisper.cpp server."""
    # Note to self: whisper is a separate process that has to be started by
    # hand (README, setup step 3). When this starts failing, it is almost
    # never the code — the server just isn't running.
    from . import config

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
