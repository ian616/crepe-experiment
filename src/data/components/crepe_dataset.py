import os
import csv
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class CREPEDataSet(Dataset):
    def __init__(self, audio_dir, annotation_dir, sample_rate=16000, transform=None):
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.sample_rate = sample_rate
        self.transform = transform

        self.frame_size = 1024
        self.hop_size = 160  # 약 10ms 간격
        self.f0_hop_time = 128 / 44100  # annotation은 약 2.9ms 간격

        self.file_names = sorted([
            f for f in os.listdir(audio_dir) if f.endswith(".wav") and not f.startswith("._")
        ])

        # 각 파일의 길이(샘플 수)만 읽어서 index_mapping 생성
        self.index_mapping = []
        for file_name in self.file_names:
            wav_path = os.path.join(audio_dir, file_name)
            info = torchaudio.info(wav_path)  # 메타데이터만 읽음
            num_samples = info.num_frames

            # 샘플레이트가 다르면 변환 후 길이 예측
            sr = info.sample_rate
            if sr != sample_rate:
                num_samples = int(num_samples * sample_rate / sr)

            num_frames = (num_samples - self.frame_size) // self.hop_size
            for i in range(num_frames):
                self.index_mapping.append((file_name, i * self.hop_size)) 

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        file_name, start = self.index_mapping[idx]
        wav_path = os.path.join(self.audio_dir, file_name)
        csv_path = os.path.join(self.annotation_dir, file_name.replace(".wav", ".csv"))

        waveform, sr = torchaudio.load(wav_path,  backend="soundfile")
        if sr != self.sample_rate:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resample(waveform)
        waveform = waveform[0]  # mono

        audio_frame = waveform[start:start + self.frame_size]
        if self.transform:
            audio_frame = self.transform(audio_frame)

        # 중심 시간에 해당하는 f0 값 추출
        center_sample = start + self.frame_size // 2
        center_time = center_sample / self.sample_rate
        f0_index = int(round(center_time / self.f0_hop_time))


        f0_data = pd.read_csv(csv_path, header=None, dtype=float).values[:, 1]

        if f0_index >= len(f0_data):  # 경계 처리
            f0 = 0.0
        else:
            f0 = f0_data[f0_index]

        target = self.create_crepe_target(f0)
        return audio_frame, target

    def hz_to_cents(self, f):
        return 1200 * np.log2(f / 10.0)

    def create_crepe_target(self, f0):

        # f0가 0이면 모든 bin이 0인 벡터 반환
        if f0 <= 0:
            return torch.zeros(360, dtype=torch.float32)
    
        # 32.70 Hz ~ 1975.5 Hz를 20 cents 간격으로 커버
        c_min = self.hz_to_cents(32.70)
        c_max = self.hz_to_cents(1975.5)
        cent_bins = np.linspace(c_min, c_max, 360)
        
        # f0 -> cent 변환
        c_true = self.hz_to_cents(f0)
        
        # Gaussian soft target
        sigma = 25
        y = np.exp(-0.5 * ((cent_bins - c_true) / sigma) ** 2)
        
        return torch.tensor(y, dtype=torch.float32)