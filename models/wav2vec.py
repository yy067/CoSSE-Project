# -*- coding : utf-8 -*-
# @FileName  : wav2vec.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Oct 12, 2023
# @Github    : https://github.com/songrise
# @Description: wav2vec2 model
import os

os.environ["HF_HOME"] = "/root/autodl-tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/.cache"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
    Wav2Vec2Config,
)
import librosa


class Wav2Vec2(nn.Module):
    def __init__(self) -> None:
        super(Wav2Vec2, self).__init__()
        # 初始化处理器（用于音频预处理）
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        # 加载配置
        self.cfg = Wav2Vec2Config()
        # 加载预训练模型
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h", config=self.cfg
        )
        # 将模型移到 GPU
        self.wav2vec2.to("cuda")
        # 池化层：将变长的音频特征池化为固定长度
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, return_features=False):
        """
        x: 原始音频波形数据
        return_features: 是否返回特征（为兼容性，返回两个相同值）
        """
        # 音频预处理
        a_feat = self.processor(
            x, return_tensors="pt", sampling_rate=16000, padding=True
        ).input_values.to("cuda")
        # cast to model precision
        # 转换为模型的精度（fp32 或 fp16）
        a_feat = a_feat.to(self.wav2vec2.dtype)
        # 通过 Wav2Vec2 模型
        a_embed = self.wav2vec2(a_feat[0])["last_hidden_state"] # 输出形状: [batch_size, sequence_length, hidden_size=768]
        # 全局平均池化得到固定长度的特征
        a_pooled = self.pooler(a_embed.transpose(1, 2)).squeeze(2)
        # 返回结果
        if return_features:
            return a_pooled, a_pooled
        return a_pooled

    def freeze_backbone(self):
        """
        冻结 Wav2Vec2 主干网络
        用于参数高效微调
        """
        self.wav2vec2.requires_grad = False


if __name__ == "__main__":
    # 创建模型
    wav2vec2 = Wav2Vec2()
    # 加载音频文件
    wav_path = "/root/autodl-tmp/mmsd/mmsd_raw_data/utterances_final/1_60.wav"
    wav_file = librosa.load(wav_path, sr=16000)[0]
    wav_file = torch.tensor(wav_file).unsqueeze(0).to("cuda")
    # 打印输入信息
    print(wav_file) # 音频波形
    print(wav_file.shape) # 形状 [1, audio_samples]
    # 前向传播
    wav2vec2_out = wav2vec2(wav_file)
    print(wav2vec2_out) # 输出的特征向量
    print(wav2vec2_out.shape) # 形状 [1, 768]
