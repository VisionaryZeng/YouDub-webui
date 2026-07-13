import os
from pathlib import Path

# 【新增】：根据报错提示，在导入 PyTorch 之前设置此环境变量，大幅减少显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import io
import librosa
import asyncio
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from nanovllm_voxcpm import VoxCPM

class VoxCPMDubbingEngine:
    """
    VoxCPM 异步配音引擎服务类
    负责底层模型的生命周期管理、并发推理以及 CPU 密集型任务的线程池调度。
    """

    def __init__(self, model_path: Path,  min_reference_ms: int, cfg_value: float, max_seqs=12, gpu_devices=[0]):
        self.model_path = model_path,
        self._is_ready = False
        self.encoder_sr = None
        self.min_reference_ms = min_reference_ms
        self.cfg_value = cfg_value

        # 初始化核心引擎 (纯计算节点，关闭缓存)
        print("⚙️ 正在初始化 VoxCPM 底层引擎...")
        self.server = VoxCPM.from_pretrained(
            model=self.model_path,
            max_num_seqs=max_seqs,
            devices=gpu_devices,
            gpu_memory_utilization=0.70,
            enable_prefix_caching=False
        )

    async def startup(self):
        """服务启动钩子：等待引擎预热并获取模型配置"""
        if self._is_ready:
            return
        await self.server.wait_for_ready()
        model_info = await self.server.get_model_info()
        self.encoder_sr = int(model_info["encoder_sample_rate"])
        self._is_ready = True
        print("✅ VoxCPM 引擎已完全就绪！")

    # ---------------------------------------------------------
    # 私有方法：纯 CPU 密集型与 I/O 操作 (需丢入线程池)
    # ---------------------------------------------------------
    @staticmethod
    def _sync_validate_and_read(file_path: str, encoder_sr: int, min_reference_ms: int):
        """校验并读取原音频，低于最小时长则拦截"""
        if not file_path or not str(file_path).strip() or not os.path.exists(file_path):
            return None, "File Not Found"

        if len(AudioSegment.from_file(file_path)) < min_reference_ms :
            return None, "Too Short"

        ref_wav, _ = librosa.load(file_path, sr=encoder_sr, mono=True)
        buffer = io.BytesIO()
        sf.write(buffer, ref_wav, encoder_sr, format="wav")
        return buffer.getvalue(), "Valid"

    @staticmethod
    def _sync_save(output_file: str, wav_data: np.ndarray, sample_rate: int = 48000):
        """保存生成的音频"""
        sf.write(output_file, wav_data, samplerate=sample_rate)

    # ---------------------------------------------------------
    # 公开 API
    # ---------------------------------------------------------
    async def extract_latents(self, audio_path: str):
        """提供给外部调用的特征提取接口 (比如用来提前生成兜底特征)"""
        if not self._is_ready:
            raise RuntimeError("Engine not started. Call await startup() first.")

        ref_wav_bytes, status = await asyncio.to_thread(
            self._sync_validate_and_read, audio_path, self.encoder_sr, self.min_reference_ms
        )
        if ref_wav_bytes is None:
            raise ValueError(f"无法提取特征: {status}")

        return await self.server.encode_latents(wav=ref_wav_bytes, wav_format="wav")

    async def process_batch(self, task_list: list[dict], fallback_latents_dict: dict) -> dict:
        """
        处理批量任务列表。
        """
        if not self._is_ready:
            raise RuntimeError("Engine not started.")

        print(f"📦 收到批量配音任务，共计 {len(task_list)} 条。")

        # 【核心保护机制】：并发信号量
        # 虽然 GPU (vLLM) 内部有队列不怕任务多，但 CPU 同时去读几百个音频文件会导致系统 I/O 崩溃。
        # 这里限制 CPU 并发读取/写入的最大数量为 24（你可以根据服务器 CPU 核数调整）。
        io_semaphore = asyncio.Semaphore(24)

        async def bounded_generate(task: dict):
            async with io_semaphore:
                success = await self.generate_dubbing(
                    task=task,
                    fallback_latents_dict=fallback_latents_dict
                )
                return {
                    "task_id": task.get("task_id"),
                    "status": "success" if success else "failed",
                    "tts_path": task.get("tts_path")
                }

        # 将所有字典任务转换为协程并打包发送给 asyncio 执行
        coroutines = [bounded_generate(task) for task in task_list]

        # 并发等待所有任务完成
        results = await asyncio.gather(*coroutines)

        # 统计成功与失败的数量
        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"🎉 批量任务执行完毕！成功: {success_count}/{len(task_list)}")

        return {
            "total": len(task_list),
            "success": success_count,
            "failed": len(task_list) - success_count,
            "details": results
        }

    async def generate_dubbing(
            self,
            task: dict,
            fallback_latents_dict: dict,
    ) -> bool:
        """
        核心配音生成接口
        """
        if not self._is_ready:
            raise RuntimeError("Engine not started.")

        # 1. 在函数内部进行按需解包，保持外部调用极简
        task_id = task.get("task_id", "unknown")
        text = task.get("dst", "")
        speaker_id = task.get("speaker")
        original_voice_path = task.get("vocals_path")
        output_file = task.get("tts_path")

        task_latents = None

        # 2. 第一优先级：尝试提取当前切片专属特征
        ref_wav_bytes, status = await asyncio.to_thread(
            self._sync_validate_and_read, original_voice_path, self.encoder_sr
        )

        if ref_wav_bytes is not None:
            task_latents = await self.server.encode_latents(wav=ref_wav_bytes, wav_format="wav")
            print(f"[{task_id}] 🎯 命中优先级 1: 成功提取原片切片特征")
        else:
            # 3. 级联降级逻辑
            if speaker_id and speaker_id in fallback_latents_dict:
                task_latents = fallback_latents_dict[speaker_id]
                print(f"[{task_id}] ⚠️ 切片无效({status})，命中优先级 2: 使用角色 [{speaker_id}] 兜底")
            elif "global" in fallback_latents_dict:
                task_latents = fallback_latents_dict["global"]
                print(f"[{task_id}] 🚨 命中优先级 3: 使用 Global 全局兜底")
            else:
                print(f"[{task_id}] ❌ 致命错误：无可用特征且无全局兜底配置！")
                return False

        # 2. 推理流
        chunks = [
            chunk async for chunk in self.server.generate(
                target_text=text,
                ref_audio_latents=task_latents,
                cfg_value=self.cfg_value
            )
            if chunk is not None
        ]

        # 3. 异步落盘
        if chunks:
            wav = np.concatenate(chunks, axis=0)
            await asyncio.to_thread(self._sync_save, output_file, wav, 48000)
            return True

        return False