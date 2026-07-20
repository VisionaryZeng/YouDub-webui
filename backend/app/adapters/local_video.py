from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import ffmpeg_binary
from ..sanitize import sanitize_text
from ..sources import SourceConfig
from ..youtube import is_local_path_url, local_path_info, local_upload_task_id


def upload_dir(workfolder: Path, task_id: str) -> Path:
    return workfolder / "_uploads" / task_id


def uploaded_video_dir(workfolder: Path, task_id: str) -> Path:
    return upload_dir(workfolder, task_id) / "video"


def remove_upload(workfolder: Path, task_id: str) -> None:
    target = upload_dir(workfolder, task_id)
    if target.exists():
        shutil.rmtree(target)


def _single_file(root: Path, task_id: str, label: str) -> Path:
    files = sorted(path for path in root.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(f"Local upload {label} is missing for task {task_id}.")
    if len(files) > 1:
        raise RuntimeError(f"Local upload has multiple {label} files for task {task_id}.")
    return files[0]


def _uploaded_video_file(workfolder: Path, task_id: str) -> Path:
    root = upload_dir(workfolder, task_id)
    if not root.exists():
        raise FileNotFoundError(f"Local upload is missing for task {task_id}.")

    video_root = uploaded_video_dir(workfolder, task_id)
    if video_root.exists():
        return _single_file(video_root, task_id, "video")
    return _single_file(root, task_id, "video")


def _title_from_url(url: str, source_file: Path) -> str:
    query = parse_qs(urlparse(url.strip()).query)
    filename = (query.get("filename") or [""])[0].strip()
    if filename:
        return Path(filename).stem or source_file.stem
    return source_file.stem


def _session_path(workfolder: Path, task_id: str, title: str) -> Path:
    safe_title = sanitize_text(title) or "local-video"
    return workfolder / "local" / f"{safe_title}__{task_id}"


def _get_stream_codecs(source_file: Path) -> tuple[str | None, str | None]:
    """检测视频和音频编码格式"""
    result = subprocess.run(
        [
            ffmpeg_binary().replace("ffmpeg", "ffprobe").replace("FFmpeg", "FFprobe"),
            "-v", "error",
            "-show_entries", "stream=codec_type,codec_name",
            "-of", "json",
            str(source_file),
        ],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    video_codec = audio_codec = None
    for stream in data.get("streams", []):
        # 只取第一个视频流和第一个音频流
        if stream.get("codec_type") == "video" and video_codec is None:
            video_codec = stream.get("codec_name")
        elif stream.get("codec_type") == "audio" and audio_codec is None:
            audio_codec = stream.get("codec_name")
    return video_codec, audio_codec


def _transcode_to_mp4(source_file: Path, video_file: Path) -> None:
    """智能转码：h264+aac 直接复制，否则重新编码"""
    video_codec, audio_codec = _get_stream_codecs(source_file)
    print(f"视频编码： {video_codec} {audio_codec}")
    # 如果已经是 h264+aac，直接复制（秒级完成）
    if video_codec == "h264" and audio_codec == "aac":
        subprocess.run(
            [
                ffmpeg_binary(),
                "-y",
                "-i",
                str(source_file),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(video_file),
            ],
            check=True,
        )
        return

    # 需要转码
    subprocess.run(
        [
            ffmpeg_binary(),
            "-y",
            "-i",
            str(source_file),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(video_file),
        ],
        check=True,
    )


def import_local_video(url: str, workfolder: Path, source: SourceConfig) -> tuple[Path, dict]:
    from .local_subtitles import uploaded_subtitle_file

    # Handle local file path URLs
    if is_local_path_url(url):
        return import_local_path(url, workfolder, source)

    task_id = local_upload_task_id(url)
    if not task_id:
        raise ValueError("Invalid local upload URL.")

    source_file = _uploaded_video_file(workfolder, task_id)
    subtitle_file = uploaded_subtitle_file(workfolder, task_id)
    title = _title_from_url(url, source_file)
    session = _session_path(workfolder, task_id, title)
    media_dir = session / "media"
    metadata_dir = session / "metadata"
    media_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    video_file = media_dir / "video_source.mp4"
    info = {
        "id": task_id,
        "title": title,
        "source": "local",
        "webpage_url": url,
        "original_path": str(source_file),
        "asr_language": source.asr_language,
        "target_language": source.target_language,
    }
    if subtitle_file:
        info["subtitle_path"] = str(subtitle_file)
    metadata_file = metadata_dir / "local_info.json"
    metadata_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    if video_file.exists() and video_file.stat().st_size > 0:
        return session, info

    _transcode_to_mp4(source_file, video_file)
    if not video_file.exists() or video_file.stat().st_size == 0:
        raise RuntimeError("ffmpeg finished without producing media/video_source.mp4")
    return session, info


def import_local_path(url: str, workfolder: Path, source: SourceConfig) -> tuple[Path, dict]:
    """Import a local file directly by path without uploading."""
    import uuid

    info = local_path_info(url)
    file_path = info.get("file")
    if not file_path:
        raise ValueError("Missing file path in local://path URL")

    source_file = Path(file_path)
    if not source_file.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")
    if not source_file.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    subtitle_path = info.get("subtitle")
    subtitle_file = None
    if subtitle_path:
        subtitle_file = Path(subtitle_path)
        if not subtitle_file.exists():
            raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

    task_id = str(uuid.uuid4())
    title = source_file.stem
    session = workfolder / "local-path" / f"{sanitize_text(title) or 'video'}__{task_id}"
    media_dir = session / "media"
    metadata_dir = session / "metadata"
    media_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 默认视频 为 mp4 格式
    media_file = media_dir / "video_source.mp4"

    if source_file.suffix == ".wav":
        media_file = media_dir / "audio_vocals.wav"
    meta = {
        "id": task_id,
        "title": title,
        "source": "local-path",
        "webpage_url": url,
        "original_path": str(source_file.absolute()),
        "asr_language": source.asr_language,
        "target_language": source.target_language,
    }
    if subtitle_path and subtitle_file:
        meta["subtitle_path"] = str(subtitle_file.absolute())
    metadata_file = metadata_dir / "local_info.json"
    metadata_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if media_file.exists() and media_file.stat().st_size > 0:
        return session, meta

    if media_file.exists() and media_file.stat().st_size > 0:
        return session, info
    if source_file.suffix == ".mp4":
        _transcode_to_mp4(source_file, media_file)

    if source_file.suffix == ".wav":
        shutil.copy2(source_file, media_file)

    if not media_file.exists() or media_file.stat().st_size == 0:
        raise RuntimeError("ffmpeg finished without producing media/video_source.mp4")
    return session, meta
