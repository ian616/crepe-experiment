from pathlib import Path

import pytest
import torch

from src.data.crepe_datamodule import CREPEDataModule


@pytest.mark.parametrize("batch_size", [32])
def test_crepe_datamodule(batch_size: int) -> None:
    audio_dir = "/mnt/HDD5/intern_crepe/MDB-stem-synth/audio_stems"
    annotation_dir = "/mnt/HDD5/intern_crepe/MDB-stem-synth/annotation_stems"

    dm = CREPEDataModule(
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        batch_size=batch_size
    )

    dm.setup()

    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    # 배치 확인
    batch = next(iter(dm.train_dataloader()))
    x, y = batch  # x: audio waveform chunk, y: pitch label

    print("\n=== Debug Sample ===")
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    print("First waveform chunk:", x[0][:10])  # 첫 오디오 chunk 앞 10개 샘플
    print("First pitch label:", y[0])

    assert len(x) == batch_size
    assert len(y) == batch_size
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32  # waveform
    assert y.dtype in (torch.float32, torch.int64)  # pitch target 타입에 맞게