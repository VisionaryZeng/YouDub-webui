import os
import io
from pathlib import Path

import torch
import torchaudio
import onnxruntime as ort
import numpy as np


class ONNXDemucsAdapter:
    def __init__(self, model_path: str, device_id: int = 0):
        """
        初始化适配器，将 FP16 的 ONNX 模型加载到 T4 GPU
        """
        self.model_path = model_path
        self.device = torch.device(f"cuda:{device_id}")
        self.target_sr = 44100
        self.n_fft = 4096
        self.hop_length = 1024

        # ⚠️ 迎合 ONNX 静态计算图：强制要求精确的 343980 采样点 (7.8 秒)
        self.chunk_size = 343980
        # 设定重叠区域为 2 秒，用于平滑过渡防止爆音
        self.overlap_size = self.target_sr * 1
        self.stride = self.chunk_size - self.overlap_size

        print("🚀 [Adapter] 正在加载 htdemucs_ft_vocals_fp16weights.onnx 到 Tensor Cores...")
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }),
            "CPUExecutionProvider"
        ]
        self.ort_session = ort.InferenceSession(self.model_path, providers=providers)

    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        陷阱 1 & 2 终结者：处理采样率和声道问题
        """
        waveform, sr = torchaudio.load(audio_path)

        # 强制重采样至 44100Hz
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)

        # 强制转换为双声道
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]  # 粗暴截取，或进行下混 (downmix)

        return waveform

    def _infer_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        # 1. 补充 Batch 维度 [1, 2, 343980]
        chunk_np = chunk.unsqueeze(0).numpy().astype(np.float32)

        # 2. 喂给 ONNX Runtime
        ort_inputs = {self.ort_session.get_inputs()[0].name: chunk_np}
        ort_outs = self.ort_session.run(None, ort_inputs)

        out_tensor = torch.from_numpy(ort_outs[0])

        # 3. 解析模型输出 (修复：精准定位人声轨)
        if out_tensor.dim() == 4:
            # 标准 Demucs 输出 4 轨: [batch, sources, channels, length]
            # 索引规则: 0=鼓, 1=贝斯, 2=其他, 3=人声
            # 动态判断: 也有特殊微调版只输出 2 轨 (0=伴奏, 1=人声)
            vocals_idx = 3 if out_tensor.shape[1] == 4 else (out_tensor.shape[1] - 1)
            vocals_waveform = out_tensor[0, vocals_idx, :, :]
        elif out_tensor.dim() == 3:
            vocals_waveform = out_tensor[0, :, :]
        else:
            vocals_waveform = out_tensor.squeeze()

        return vocals_waveform

    def separate_vocals(self, input_path: str, output_path: str) -> Path:
        waveform = self._preprocess_audio(input_path)
        total_length = waveform.shape[1]

        final_vocals = torch.zeros_like(waveform)
        weight_sum = torch.zeros(total_length)

        # 【核心修复】：构建梯形平顶窗 (Trapezoidal Window)
        # 前 overlap_size: 线性淡入 (0 -> 1)
        # 中间区域: 保持满音量 1.0 (保证声音平稳不颤抖)
        # 后 overlap_size: 线性淡出 (1 -> 0)
        window = torch.ones(self.chunk_size)
        window[:self.overlap_size] = torch.linspace(0, 1, self.overlap_size)
        window[-self.overlap_size:] = torch.linspace(1, 0, self.overlap_size)

        print(f"🎵 [Adapter] 开始分离人声，总长: {total_length / self.target_sr:.2f} 秒...")

        for i in range(0, total_length, self.stride):
            end_idx = min(i + self.chunk_size, total_length)
            chunk = waveform[:, i:end_idx]
            current_chunk_length = chunk.shape[1]

            # 尾部补零
            if current_chunk_length < self.chunk_size:
                pad_size = self.chunk_size - current_chunk_length
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))

            # 推理获取人声
            vocal_chunk = self._infer_chunk(chunk)

            # 截取有效部分，应用梯形窗
            vocal_chunk = vocal_chunk[:, :current_chunk_length]
            current_window = window[:current_chunk_length].to(vocal_chunk.device)

            # 累加音频和权重
            final_vocals[:, i:end_idx] += vocal_chunk * current_window
            weight_sum[i:end_idx] += current_window

        # 防止边缘除以 0 导致杂音，进行归一化
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        final_vocals = final_vocals / weight_sum

        # 落盘
        torchaudio.save(output_path, final_vocals, self.target_sr, format="wav")
        print(f"✅ [Adapter] 人声提取完成: {output_path}")

        return Path(output_path)