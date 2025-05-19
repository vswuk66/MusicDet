import torch
import torchaudio
import os
from pathlib import Path
import json

# Audio preprocessing parameters (from config.py)
SAMPLE_RATE = 22050
DURATION = 30  # seconds
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
FMIN = 0
FMAX = None
TARGET_LENGTH = 128

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=FMIN,
    f_max=FMAX or SAMPLE_RATE // 2
)

# === Genre mapping ===
GENRE_MAPPING_PATH = Path("genre_mapping.json")
def load_genre_mapping():
    if GENRE_MAPPING_PATH.exists():
        with open(GENRE_MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # Преобразуем строковые ключи в числа
        return {int(k): v for k, v in mapping.items()}
    # fallback: дефолтный mapping (может не совпадать с обучающим!)
    return {
        0: 'Electronic',
        1: 'Experimental',
        2: 'Folk',
        3: 'Hip-Hop',
        4: 'Instrumental',
        5: 'International',
        6: 'Pop',
        7: 'Rock',
        8: 'Classical',
        9: 'Jazz',
        10: 'Old-Time / Historic',
        11: 'Soul-RnB',
        12: 'Spoken',
        13: 'Blues',
        14: 'Country',
        15: 'Easy Listening'
    }

def save_genre_mapping(mapping):
    # Преобразуем числовые ключи в строки для JSON
    mapping_str = {str(k): v for k, v in mapping.items()}
    with open(GENRE_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping_str, f, ensure_ascii=False, indent=2)

GENRE_MAPPING = load_genre_mapping()
REVERSE_GENRE_MAPPING = {v: k for k, v in GENRE_MAPPING.items()}

def extract_features(audio_path):
    """
    Extract mel spectrogram features from audio file.
    Returns tensor of shape [1, 1, 128, 128]
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
    cur_len = mel_spec.shape[-1]
    if cur_len > TARGET_LENGTH:
        start = (cur_len - TARGET_LENGTH) // 2
        mel_spec = mel_spec[:, :, start:start + TARGET_LENGTH]
    elif cur_len < TARGET_LENGTH:
        pad = TARGET_LENGTH - cur_len
        mel_spec = torch.nn.functional.pad(mel_spec, (0, pad))
    if mel_spec.shape != (1, 128, 128):
        raise ValueError(f"Spectrogram shape error: {mel_spec.shape}")
    mel_spec = 2 * (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-10) - 1
    return mel_spec.unsqueeze(0)

def get_genre_name(prediction):
    """
    Convert numeric prediction to genre name
    """
    return REVERSE_GENRE_MAPPING.get(prediction, "Unknown")
