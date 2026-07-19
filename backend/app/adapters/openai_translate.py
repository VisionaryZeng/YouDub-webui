from __future__ import annotations

import itertools
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from google.auth import default
import google.auth.transport.requests
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..sources import SourceConfig
from ._translate_prompts import PROMPT_DICT
from .openai_client import normalize_openai_base_url

log = logging.getLogger(__name__)

API_SETTING_KEYS = ("base_url", "api_key", "model")
CHUNK_SIZE = 10
DESCRIPTION_LIMIT = 500
DEFAULT_CONCURRENCY = 50


class HotWordItem(BaseModel):
    src: str = Field(description="原文术语")
    dst: str = Field(description="目标语言推荐译法；如 Transformer/GPU 一类应保持原样，则 dst 与 src 相同")


class CorrectionItem(BaseModel):
    wrong: str = Field(description="转录中明显错认的写法")
    correct: str = Field(description="正确写法")


class PreprocessResponse(BaseModel):
    summary: str = Field(default="", description="用目标译文语言写的视频摘要，长度限制 3-5 句")
    hotwords: list[HotWordItem] = Field(default_factory=list, description="热词识别列表")
    corrections: list[CorrectionItem] = Field(default_factory=list, description="ASR 纠错要点列表")

class TranslationResBase(BaseModel):
    subtitle_list: list[str] = Field(default_factory=list, description="按顺序翻译好的句子列表")

def get_translation_type(target_length: int) -> type[TranslationResBase]:
    class TranslationRes(TranslationResBase):
        # 第一个参数必须是 cls，代表 TranslationRes 这个类
        @field_validator('subtitle_list')
        def check_length(cls, v):
            if len(v) != target_length:
                # 这个报错信息会被 instructor 自动抓取，并塞回给 LLM 让他重试
                raise ValueError(f"列表长度必须是 {target_length}，但你返回了 {len(v)}")
            return v

        # 校验 2：检查是否包含空字符串
        @field_validator('subtitle_list')
        def check_no_empty_strings(cls, v):
            # 遍历列表，如果发现去除空格后是空的，就报错
            if any(not item.strip() for item in v):
                raise ValueError("返回的列表中包含了空字符串，这是不允许的")
            return v

    # 返回这个刚刚“捏”好的类（注意是返回类本身，不是类的实例）
    return TranslationRes


def list_models(*, base_url: str, api_key: str) -> list[str]:
    if not api_key:
        raise ValueError("OpenAI API key is not configured.")
    client = OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url))
    response = client.models.list()
    seen: set[str] = set()
    models: list[str] = []
    for item in response.data:
        model_id = getattr(item, "id", "")
        if model_id and model_id not in seen:
            seen.add(model_id)
            models.append(model_id)
    return models


def _client(base_url: str, api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("OpenAI API key is not configured.")

    # client = OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url),max_retries = 2)

    # 1. 动态获取 Google Cloud 认证 Token
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())

    client = OpenAI(
        api_key=credentials.token,
        base_url="https://us-central1-aiplatform.googleapis.com/v1beta1/projects/project-4e4de0ce-a156-416f-bad/locations/us-central1/endpoints/openapi",
        max_retries=2
    )
    return instructor.patch(client)



def _call_json[T: BaseModel](client: OpenAI, model: str, system: str, user: str, schema: type[T]) -> T:
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
        temperature=0.2,
    )
    return response.choices[0].message.parsed



def _meta_view(meta: dict[str, Any]) -> dict[str, str]:
    description = (meta.get("description") or "").strip()
    if len(description) > DESCRIPTION_LIMIT:
        description = description[:DESCRIPTION_LIMIT] + "..."
    return {
        "title": str(meta.get("title") or "").strip() or "(unknown)",
        "uploader": str(meta.get("uploader") or "").strip() or "(unknown)",
        "description": description or "(none)",
    }


def preprocess(
    full_text: str,
    meta: dict[str, Any],
    source: SourceConfig,
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> PreprocessResponse:

    system = PROMPT_DICT[source.target_language]["system_preprocess"].format(
        src_language_name=source.asr_language,
        dst_language_name=source.target_language_name,
        **_meta_view(meta),
    )
    user = PROMPT_DICT[source.target_language]["user_preprocess"].format(full_text=full_text)
    client = _client(base_url, api_key)
    try:
        data = _call_json(client, model, system, user, PreprocessResponse)
        return data
    except (json.JSONDecodeError, ValidationError) as exc:
        log.error("preprocess failed: %s", exc)
    return PreprocessResponse()


def _post_process(lines: list[str], target_language: str) -> list[str]:
    return [
        line.strip().replace("——", "，") if target_language == "zh" else line.strip()
        for line in lines
    ]


def translate_sentence(
    lines: list[str],
    source: SourceConfig,
    client: OpenAI,
    model: str,
    system: str,
) -> list[str]:
    # 1. 拼接成完整上下文
    context_sentence = " ".join(lines)
    # 2. 将数组转为 JSON 格式字符串，展示原本的“形状”
    chunks_json = json.dumps(lines, ensure_ascii=False, indent=2)

    user = PROMPT_DICT[source.target_language]["user_translate"].format(context_sentence=context_sentence, chunks_json = chunks_json)
    translation_type = get_translation_type(len(lines))
    try:
        data = _call_json(client, model, system, user, translation_type)
        return _post_process(data.subtitle_list, source.target_language)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        last_error = exc
        log.error("translate attempt failed for %r: %s", lines[:60], exc)
    raise RuntimeError(f"translate_sentence failed : {last_error}")


def translate_batch(
    texts: list[str],
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[str]:
    if not texts:
        return []

    system = PROMPT_DICT[source.target_language]["system_translate"].format(
        src_language_name=source.asr_language,
        dst_language_name=source.target_language_name,
        summary=pre.summary,
        hotwords=pre.hotwords,
        corrections=pre.corrections,
        **_meta_view(meta),
    )
    client = _client(base_url, api_key)
    log.info(
        "translate_batch: %d sentences, concurrency=%d", len(texts), concurrency,
    )

    chunks = [texts[i:i + CHUNK_SIZE] for i in range(0, len(texts), CHUNK_SIZE)]

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        # 1. 拿到包含子列表的迭代器
        chunked_results = pool.map(
            lambda chunk: translate_sentence(chunk, source, client, model, system),
            chunks,
        )
        # 2. 一键拍平并转换为标准的一维 list
        return list(itertools.chain.from_iterable(chunked_results))


def _read_meta(session: Path) -> dict[str, Any]:
    info_file = session / "metadata" / "ytdlp_info.json"
    if not info_file.exists():
        return {}
    return json.loads(info_file.read_text(encoding="utf-8"))


def _speaker(utt: dict[str, Any]) -> str:
    additions = utt.get("additions") or {}
    if isinstance(additions, dict):
        return str(additions.get("speaker") or "1")
    return "1"


def _full_text(data: dict[str, Any], texts: list[str]) -> str:
    raw = data.get("result", {}).get("text") or ""
    if raw.strip():
        return raw
    return " ".join(texts)


def preprocess_artifact_path(session: Path) -> Path:
    return session / "metadata" / "translation_preprocess.json"


def write_preprocess_artifact(session: Path, pre: PreprocessResponse) -> Path:
    path = preprocess_artifact_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pre.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_preprocess_artifact(session: Path) -> PreprocessResponse | None:
    path = preprocess_artifact_path(session)
    if not path.exists():
        return None
    return PreprocessResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _concurrency_from(settings: dict[str, str]) -> int:
    raw = str(settings.get("translate_concurrency") or "").strip()
    if not raw or not all("0" <= char <= "9" for char in raw):
        return DEFAULT_CONCURRENCY
    concurrency = int(raw)
    if concurrency < 1 or concurrency > 200:
        return DEFAULT_CONCURRENCY
    return concurrency


def translate_asr(
    asr_file: Path,
    session: Path,
    settings: dict[str, str],
    source: SourceConfig,
) -> Path:
    output_file = session / "metadata" / f"translation.{source.target_language}.json"
    if output_file.exists():
        return output_file

    data = json.loads(asr_file.read_text(encoding="utf-8"))
    utterances = data["result"]["utterances"]
    texts = [u["text"].strip() for u in utterances]
    full_text = _full_text(data, texts)
    meta = _read_meta(session)

    api = {key: settings[key] for key in API_SETTING_KEYS if key in settings}
    pre = load_preprocess_artifact(session)
    if pre is None:
        pre = preprocess(full_text, meta, source, **api)
        write_preprocess_artifact(session, pre)
        log.info("Wrote translation preprocess artifact to %s", preprocess_artifact_path(session))
    else:
        log.info("Reusing translation preprocess artifact from %s", preprocess_artifact_path(session))
    dst_list = translate_batch(
        texts, source, meta, pre, **api, concurrency=_concurrency_from(settings)
    )

    translation = [
        {
            "src": text,
            "dst": dst,
            "src_lang": source.asr_language,
            "dst_lang": source.target_language,
            "start_time": utt["start_time"],
            "end_time": utt["end_time"],
            "speaker": _speaker(utt),
        }
        for text, dst, utt in zip(texts, dst_list, utterances)
    ]
    output_file.write_text(
        json.dumps({"translation": translation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    srt_file = session / "metadata" / f"subtitles.bilingual.srt"
    lines: list[str] = []

    for idx, item in enumerate(translation, start=1):
        lines.extend([str(idx), f"{_srt_time(item["start_time"])} --> {_srt_time(item["end_time"])}", item["src"], item["dst"], ""])

    srt_file.write_text("\n".join(lines), encoding="utf-8")

    return output_file


def _srt_time(ms: int) -> str:
    hours = ms // 3_600_000
    ms -= hours * 3_600_000
    minutes = ms // 60_000
    ms -= minutes * 60_000
    seconds = ms // 1000
    millis = ms - seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"