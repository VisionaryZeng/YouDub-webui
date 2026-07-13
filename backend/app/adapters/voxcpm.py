from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

import soundfile as sf
from pydub import AudioSegment

from .sync_bridge import SyncDubbingBridge
from ..config import MODEL_CACHE_DIR
from ..devices import has_gpu

_MODEL = None
MAX_CONCURRENCY = 12
GLOBA_FALLBACKL_KEY = "global_fallback_voice"
_PROMPT_CACHE_GENERATION_DEFAULTS = {
    "min_len": 2,
    "max_len": 4096,
    "retry_badcase": True,
    "retry_badcase_max_times": 3,
    "retry_badcase_ratio_threshold": 6.0,
}


def _model_path() -> Path:
    configured_dir = os.getenv("VOXCPM_MODEL_DIR")
    if configured_dir:
        return Path(configured_dir).expanduser()

    model_id = os.getenv("VOXCPM_MODEL", "OpenBMB/VoxCPM2")
    local_dir = MODEL_CACHE_DIR / model_id.replace("/", "__")
    from modelscope import snapshot_download

    downloaded = snapshot_download(model_id, local_dir=str(local_dir))
    return Path(downloaded)


def _load_model():
    global _MODEL
    if _MODEL is None:
        from voxcpm import VoxCPM

        _MODEL = VoxCPM.from_pretrained(
            str(_model_path()),
            load_denoiser=os.getenv("VOXCPM_LOAD_DENOISER", "false").lower() == "true",
        )
    return _MODEL


def _first_reference(files: list[Path], min_ms: int) -> Path | None:
    for path in files:
        if len(AudioSegment.from_file(path)) >= min_ms:
            return path
    if files:
        return files[0]
    return None


def _speaker(item: dict) -> str:
    speaker = item.get("speaker")
    if speaker is None:
        return "1"
    speaker = str(speaker).strip()
    return speaker or "1"


def _fallback_references(vocals_dir: Path, items: list[dict], min_ms: int) -> dict[str, Path]:
    files = sorted(vocals_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError("No vocal segments were generated for VoxCPM references.")

    fallbacks: dict[str, Path] = {}
    global_fallback = _first_reference(files, min_ms) or files[0]
    fallbacks[GLOBA_FALLBACKL_KEY] = global_fallback
    speaker_files: dict[str, list[Path]] = {}
    for index, item in enumerate(items, start=1):
        reference = vocals_dir / f"{index:04d}.wav"
        if reference.exists():
            speaker_files.setdefault(_speaker(item), []).append(reference)

    for speaker, refs in speaker_files.items():
        fallback = _first_reference(refs, min_ms)
        if fallback is not None:
            fallbacks[speaker] = fallback

    return fallbacks


def _tts_text(item: dict) -> str:
    text = item.get("dst") or item.get("zh", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("target text must be a non-empty string")
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text)


"""
核心参数：

text / target_text：要说的内容
reference_wav_path / prompt_cache：音色来源

质量参数：

cfg_value：稳定性 vs 多样性
inference_timesteps：质量 vs 速度

容错参数：

min_len / max_len：生成时长范围
retry_*：异常处理策略

决策参数：

min_reference_ms：决定使用哪种生成方式
"""

def generate_tts(
    translation_file: Path,
    vocals_dir: Path,
    session: Path,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Path:
    output_dir = session / "segments" / "tts"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    items = data["translation"]
    total = len(items)
    if total == 0:
        if progress_callback:
            progress_callback(100, "No TTS clips to generate")
        return output_dir

    min_reference_ms = int(os.getenv("VOXCPM_MIN_REFERENCE_MS", "1200"))
    fallback_references= _fallback_references(vocals_dir, items, min_reference_ms)
    cfg_value = float(os.getenv("VOXCPM_CFG_VALUE", "2.0"))
    inference_timesteps = int(os.getenv("VOXCPM_INFERENCE_TIMESTEPS", "10"))

    if has_gpu():
        async_generate_tts(cfg_value, fallback_references, items, min_reference_ms, output_dir, vocals_dir)
    else:
        sync_generate_tts(items, fallback_references, cfg_value, inference_timesteps, min_reference_ms, vocals_dir, output_dir, progress_callback)

    return output_dir


def async_generate_tts(cfg_value: float, fallback_references: dict[str, Path], items, min_reference_ms: int, output_dir: Path, vocals_dir: Path):
    dubbing_service = SyncDubbingBridge(
        model_path=str(_model_path()),
        max_seqs=MAX_CONCURRENCY,
        min_reference_ms=min_reference_ms,
        cfg_value=cfg_value
    )

    fallback_references_latents = {
        key: dubbing_service.extract_latents(str(val))
        for key, val in fallback_references.items()
    }

    task_dicts = []
    for idx, item in enumerate(items, start=1):
        task = dict(item)
        task["task_id"] = f"{idx:04d}"
        task["vocals_path"] = f"{vocals_dir}/{task["task_id"]}.wav"
        task["tts_path"] = f"{output_dir}/{task["task_id"]}.wav"
        task_dicts.append(task)

    summary_report = dubbing_service.process_batch(
        task_list=task_dicts,
        fallback_latents_dict=fallback_references_latents,  # 传入处理好的特征字典
    )
    print(f"task completed: total: {summary_report['total']} success: {summary_report['success']}  failed: {summary_report['failed']} details: {summary_report['details']} ")


def sync_generate_tts(items: list, fallback_references: dict[str, Path], cfg_value: float,  inference_timesteps: int, min_reference_ms: int, vocals_dir: Path, output_dir: Path, progress_callback: Callable[[int, str], None] | None):
    total = len(items)
    model = _load_model()
    fallback_caches = {}

    for index, item in enumerate(items, start=1):
        output_file = output_dir / f"{index:04d}.wav"
        if not output_file.exists():
            reference = vocals_dir / f"{index:04d}.wav"
            text = _tts_text(item)
            if not reference.exists() or len(AudioSegment.from_file(reference)) < min_reference_ms:
                speaker = _speaker(item)
                if speaker not in fallback_caches:
                    fallback = fallback_references.get(speaker, fallback_references[GLOBA_FALLBACKL_KEY])
                    fallback_caches[speaker] = model.tts_model.build_prompt_cache(
                        reference_wav_path=str(fallback)
                    )
                result = model.tts_model.generate_with_prompt_cache(
                    target_text=text,
                    prompt_cache=fallback_caches[speaker],
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    **_PROMPT_CACHE_GENERATION_DEFAULTS,
                )
                wav_tensor, _, _ = result
                wav = wav_tensor.squeeze(0).cpu().numpy()
            else:
                wav = model.generate(
                    text=text,
                    reference_wav_path=str(reference),
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                )
            sf.write(output_file, wav, model.tts_model.sample_rate)
        if progress_callback:
            progress = round(index / total * 100)
            progress_callback(progress, f"Prepared {index}/{total} TTS clips")
