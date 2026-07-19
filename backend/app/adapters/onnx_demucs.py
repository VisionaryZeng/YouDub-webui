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

        # Colab T4 16GB 显存，设定每次切块长度为 30 秒
        self.chunk_size = self.target_sr * 30
        # 设定重叠区域为 2 秒，用于平滑过渡防止爆音
        self.overlap_size = self.target_sr * 2
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
        """
        修正版：直接喂入波形，将 STFT 计算完全交给 ONNX 内部处理
        """
        # 1. chunk 原本是 [2, length] 的 2 维张量。
        # 我们使用 unsqueeze(0) 为其增加一个 Batch 维度，变成 [1, 2, length] (正好是模型期望的 Rank 3)
        chunk_np = chunk.unsqueeze(0).numpy().astype(np.float32)

        # 2. 喂给 ONNX Runtime
        ort_inputs = {self.ort_session.get_inputs()[0].name: chunk_np}
        ort_outs = self.ort_session.run(None, ort_inputs)

        # 3. 解析模型输出
        out_tensor = torch.from_numpy(ort_outs[0])

        # ONNX 的输出可能有两种情况：
        # 模式 A: [1, 2, length] (只输出人声)
        # 模式 B: [1, 1, 2, length] (标准 Demucs 结构: [batch, sources, channels, length])
        if out_tensor.dim() == 4:
            # 如果是 4 维，剥离 batch 和 source 维度
            vocals_waveform = out_tensor[0, 0, :, :]
        elif out_tensor.dim() == 3:
            # 如果是 3 维，剥离 batch 维度
            vocals_waveform = out_tensor[0, :, :]
        else:
            # 暴力降维作为最后防线
            vocals_waveform = out_tensor.squeeze()

        return vocals_waveform

    def separate_vocals(self, input_path: str, output_path: str) -> Path:
        """
        陷阱 4 终结者：核心调度接口，处理长音频并消除切块爆音 (Overlap-Add)
        """
        waveform = self._preprocess_audio(input_path)
        total_length = waveform.shape[1]

        # 预分配全长的输出张量（填满 0）和权重记录器
        final_vocals = torch.zeros_like(waveform)
        weight_sum = torch.zeros(total_length)

        # 生成 Hanning 窗，用于切块边缘的淡入淡出（防爆音黑科技）
        window = torch.hann_window(self.chunk_size)

        print(f"🎵 [Adapter] 开始分离人声，总长: {total_length / self.target_sr:.2f} 秒...")

        # 滑动窗口切块处理
        for i in range(0, total_length, self.stride):
            end_idx = min(i + self.chunk_size, total_length)
            chunk = waveform[:, i:end_idx]
            current_chunk_length = chunk.shape[1]

            # 如果到了最后一小块，且长度不足以支撑模型运算，进行补零 Padding
            if current_chunk_length < self.chunk_size:
                pad_size = self.chunk_size - current_chunk_length
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))

            # 执行 GPU 推理
            vocal_chunk = self._infer_chunk(chunk)

            # 截取有效部分（去掉可能 padding 的部分），并应用 Hanning 窗平滑边缘
            vocal_chunk = vocal_chunk[:, :current_chunk_length]
            current_window = window[:current_chunk_length].to(vocal_chunk.device)

            # Overlap-Add 重叠追加
            final_vocals[:, i:end_idx] += vocal_chunk * current_window
            weight_sum[i:end_idx] += current_window

        # 归一化重叠部分，防止重叠处声音过大
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        final_vocals = final_vocals / weight_sum

        # 落盘保存为 Whisper 需要的无损 WAV 格式
        torchaudio.save(output_path, final_vocals, self.target_sr, format="wav")
        print(f"✅ [Adapter] 人声提取完成: {output_path}")

        return Path(output_path)