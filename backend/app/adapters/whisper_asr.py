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

    import stable_whisper
    import whisper

    name = os.getenv("WHISPER_MODEL", "large-v3-turbo")
    whisper_device = resolve_device("whisper").selected
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT") or None
    try:
        _MODEL = stable_whisper.load_model(name, device=whisper_device, download_root=download_root)
    except RuntimeError as exc:
        if not _is_checksum_error(exc):
            raise
        if not _remove_corrupt_whisper_cache(whisper, name, download_root):
            raise
        _MODEL = stable_whisper.load_model(name, device=whisper_device, download_root=download_root)

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


def _convert_segments_stable(segments: list) -> list:
    full_line = []
    for seg in segments:
        line = {
            "text": "",
            "start_time": 0,
            "end_time": 0,
            "words": seg.get("words", []),
        }
        concat_words(line)
        full_line.append(line)

    return full_line

def _convert_segments(segments: list) -> list:
    line = {
        "text": "",
        "start_time": 0,
        "end_time": 0,
        "words": [],
    }
    full_line = [line]
    for seg in segments:
        for word in seg.get("words", []):
            # 0 间隔超出 1s，要拆分
            cur_words : list = line.get("words")
            if len(cur_words) > 1 and word.get("start") - cur_words[-1].get("end") >= 1.0:
                line = finish_line(line, full_line)
                cur_words = line.get("words")

            cur_words.append(word)

            # 满足超出 10 个字符时作为一行字幕，类似坐电梯，没超重就进电梯，超重了就等下一次电梯
            if word.get("word", "").rstrip().endswith((",",".","?",";")) and len(cur_words) > 10:
                # 1 标点符号： 逗号、分号、冒号是最高优先级的切分点。
                symbol_idx = rfind_delimiter((".","?",",",";"), cur_words[:-1])
                if symbol_idx != -1:
                    line["words"]= cur_words[: symbol_idx+1]
                    line = finish_line(line, full_line)
                    line["words"] = cur_words[symbol_idx + 1 :]
                    continue

                # 2 并列连词： 在 and, but, or, so 之前切分。（注意：连词应该留在下一行的开头，而不是上一行的结尾。比如：...went to the store, / but it was closed.）
                word_idx = rfind_delimiter(("and", "but", "or", "so", "however"), cur_words)
                if word_idx != -1:
                    line["words"] = cur_words[: word_idx - 1]
                    line = finish_line(line, full_line)
                    line["words"] = cur_words[word_idx:]
                    continue

                # 3 从属连词： 在 because, if, although, when 之前切分。
                word_idx = rfind_delimiter(("because", "if", "although", "when"), cur_words)
                if word_idx != -1:
                    line["words"] = cur_words[: word_idx - 1]
                    line = finish_line(line, full_line)
                    line["words"] = cur_words[word_idx:]
                    continue
                # 4 关系代词： 在定语从句的引导词 which, who, that 之前切分。
                word_idx = rfind_delimiter(("which", "who", "that"), cur_words)
                if word_idx != -1:
                    line["words"] = cur_words[: word_idx - 1]
                    line = finish_line(line, full_line)
                    line["words"] = cur_words[word_idx:]

    concat_words(line)
    return full_line

def rfind_delimiter(delimiter: tuple, words: list[dict]) -> int:
    for idx in range(len(words) - 1, -1, -1):
        word = words[idx].get("word", "")
        # 找到在 10 个 word 里面的分隔符，避免太长
        if word.rstrip().endswith(delimiter) and len(words) - idx <= 10:
            return idx

    return -1


def finish_line(line: dict[str, str], full_line: list) -> dict:
    concat_words(line)
    line = {
        "text": "",
        "start_time": 0,
        "end_time": 0,
        "words": [],
    }
    full_line.append(line)
    return line


def concat_words(line: dict):
    line["text"] = "".join(word.get("word", "") for word in line["words"]).strip()
    line["words"] = _convert_words(line["words"])
    line["start_time"] = line["words"][0].get("start_time", 0.0)
    line["end_time"] = line["words"][-1].get("end_time", 0.0)


# 假设 base_segments 是经过 stable-ts 初步转录并转为 dict 后的片段列表
# 例如你已经跑了: base_segments = result_obj.to_dict()["segments"]

def merge_short_segments(segments, max_gap_seconds=1.0, max_words=10):
    if not segments:
        return []

    merged_chunks = []

    # 初始化第一个块
    current_text = segments[0]["text"].strip()
    current_start = segments[0]["start"]
    current_end = segments[0]["end"]
    # 统计英文单词数（按空格切分）
    current_word_count = len(current_text.split())

    for next_chunk in segments[1:]:
        next_text = next_chunk["text"].strip()
        next_start = next_chunk["start"]
        next_end = next_chunk["end"]
        next_word_count = len(next_text.split())

        # 计算两句话之间的静音间隔
        gap = next_start - current_end
        # 计算如果拼接在一起的总单词数
        combined_word_count = current_word_count + next_word_count

        # 核心判断逻辑：间隔不超标 且 字数不超标
        if gap <= max_gap_seconds and combined_word_count <= max_words:
            # 允许拼接！吸收下一个片段
            current_text = current_text + " " + next_text
            current_end = next_end
            current_word_count = combined_word_count
        else:
            # 条件不满足，把当前已经吸饱的块存入结果库
            merged_chunks.append({
                "start": current_start,
                "end": current_end,
                "text": current_text
            })
            # 开启一个新的收集块
            current_text = next_text
            current_start = next_start
            current_end = next_end
            current_word_count = next_word_count

    # 循环结束后，别忘了把最后剩下的那个块收尾加进去
    merged_chunks.append({
        "start": current_start,
        "end": current_end,
        "text": current_text
    })

    return merged_chunks

def recognize_speech(vocals_file: Path, session: Path, language: str) -> Path:
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"
    if output_file.exists():
        return output_file

    model = _load_model()
    result_obj = model.transcribe(
        str(vocals_file),
        language=language,
        word_timestamps=True,
        verbose=False,
    )

    # 2. 核心魔法：在内存中对结果进行重新切分
    # 这个方法会根据时间戳和语义，智能地把过长的句子拆开，确保每个片段不超过 10 个词
    result_obj.split_by_length(max_words=10)

    # 3. 将对象转回原版 Whisper 的字典格式，保持与你原有下游代码的兼容性
    result = result_obj.to_dict()

    # utterances = _convert_segments_stable(result.get("segments", []))

    utterances = merge_short_segments(segments=result.get("segments", []), max_gap_seconds=1.0, max_words=10)

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
