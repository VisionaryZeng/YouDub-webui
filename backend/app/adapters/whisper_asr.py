from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from pydub import AudioSegment

from ..devices import resolve_device

_MODEL = None


def _whisper_cache_file(whisper, name: str, download_root: str | None) -> Path | None:
    if not download_root:
        return None
    model_url = getattr(whisper, "_MODELS", {}).get(name)
    if not model_url:
        return None
    filename = Path(urlparse(model_url).path).name
    if not filename:
        return None
    return Path(download_root).expanduser() / filename


def _is_checksum_error(exc: RuntimeError) -> bool:
    return "sha256 checksum" in str(exc).lower()


def _remove_corrupt_whisper_cache(whisper, name: str, download_root: str | None) -> bool:
    cache_file = _whisper_cache_file(whisper, name, download_root)
    if not cache_file or not cache_file.exists():
        return False
    cache_file.unlink()
    return True


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import whisper

    name = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    whisper_device = resolve_device("whisper").selected
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT") or None
    try:
        _MODEL = whisper.load_model(name, device=whisper_device, download_root=download_root)
    except RuntimeError as exc:
        if not _is_checksum_error(exc):
            raise
        if not _remove_corrupt_whisper_cache(whisper, name, download_root):
            raise
        _MODEL = whisper.load_model(name, device=whisper_device, download_root=download_root)

    return _MODEL


def _to_ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000))


def _convert_words(words: list) -> list:
    return [
        {
            "text": w.get("word", ""),
            "start_time": _to_ms(w.get("start", 0.0)),
            "end_time": _to_ms(w.get("end", 0.0)),
        }
        for w in words or []
    ]


def _convert_segments(segments: list) -> list:
    pre_line = {
        "text": "",
        "start_time": 0,
        "end_time": 0,
        "words": [],
    }
    full_line = [pre_line]
    cur_line = []
    for seg in segments:
        for word in seg.get("words", []):
            cur_line.append(word)
            # line["words"].append(word)
            if word.get("word", "").rstrip().endswith(",","."):
                # 满足超出 16 个字符时作为一行字幕，类似坐电梯，没超重就进电梯，超重了就等下一次电梯
                if len(pre_line["words"]) + len(cur_line) > 16:
                    concat_line(pre_line)
                    pre_line = {
                        "text": "",
                        "start_time": 0,
                        "end_time": 0,
                        "words": cur_line,
                    }
                    full_line.append(pre_line)
                else:
                    pre_line["words"].extend(cur_line)
                cur_line = []

    concat_line(pre_line)
    return full_line


def concat_line(pre_line: dict):
    pre_line["text"] = "".join(word.get("word", "") for word in pre_line["words"]).strip()
    pre_line["words"] = _convert_words(pre_line["words"])
    pre_line["start_time"] = pre_line["words"][0].get("start_time", 0.0)
    pre_line["end_time"] = pre_line["words"][-1].get("end_time", 0.0)


def recognize_speech(vocals_file: Path, session: Path, language: str) -> Path:
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"
    if output_file.exists():
        return output_file

    model = _load_model()
    result = model.transcribe(
        str(vocals_file),
        language=language,
        word_timestamps=True,
        verbose=False,
    )

    utterances = _convert_segments(result.get("segments", []))
    if not utterances:
        raise RuntimeError("Whisper did not return any segments.")

    duration_ms = len(AudioSegment.from_file(vocals_file))
    payload = {
        "audio_info": {"duration": duration_ms},
        "result": {
            "text": (result.get("text") or "").strip(),
            "utterances": utterances,
        },
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file
