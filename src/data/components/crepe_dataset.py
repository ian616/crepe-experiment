import os
import csv
import torch
import torchaudio
from torch.utils.data import Dataset
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
            f for f in os.listdir(audio_dir) if f.endswith(".wav")
        ])

        # 모든 (파일 이름, 시작 index 리스트)를 미리 구함
        self.index_mapping = []
        for file_name in self.file_names:
            wav_path = os.path.join(audio_dir, file_name)
            waveform, sr = torchaudio.load(wav_path)
            if sr != sample_rate:
                resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resample(waveform)
            waveform = waveform[0]  # mono

            num_frames = (len(waveform) - self.frame_size) // self.hop_size
            for i in range(num_frames):
                self.index_mapping.append((file_name, i * self.hop_size))

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        file_name, start = self.index_mapping[idx]
        wav_path = os.path.join(self.audio_dir, file_name)
        csv_path = os.path.join(self.annotation_dir, file_name.replace(".wav", ".csv"))

        waveform, sr = torchaudio.load(wav_path)
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

        f0_data = pd.read_csv(csv_path, header=None).values[:, 1]
        if f0_index >= len(f0_data):  # 경계 처리
            f0 = 0.0
        else:
            f0 = f0_data[f0_index]

        return {
            "waveform": audio_frame,
            "f0": torch.tensor(f0, dtype=torch.float32),
            "file": file_name,
            "center_time": center_time
        }