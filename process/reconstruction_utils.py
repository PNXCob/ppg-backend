import os
import numpy as np
import torch
from typing import Tuple
from sklearn import preprocessing
from scipy.signal import resample, savgol_filter, find_peaks
import more_itertools as mit
from torch import nn
from collections import deque
from sqa_advanced_utils import (
    sqa,
    bandpass_filter,
    resample_signal
)

# Paths and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")
GAN_MODEL_FILE_NAME = 'GAN_model.pth'
MAX_RECONSTRUCTION_LENGTH_SEC = 15
UPSAMPLING_RATE = 2
RECONSTRUCTION_MODEL_SAMPLING_FREQUENCY = 20

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, 32, 4, 2, 0, bias=False, padding_mode='replicate'), nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2, 0, bias=False, padding_mode='replicate'), nn.InstanceNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 0, bias=False, padding_mode='replicate'), nn.InstanceNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 128, 4, 2, 0, bias=False), nn.ReLU(), nn.Dropout(0.5),
            nn.Conv1d(128, 128, 4, 2, 1, bias=False), nn.ReLU(), nn.Dropout(0.5),
            nn.Conv1d(128, 128, 4, 2, 1, bias=False), nn.ReLU(), nn.Dropout(0.5)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 128, 4, 2, output_padding=1, bias=False), nn.InstanceNorm1d(128), nn.ReLU(True), nn.Dropout(0.5),
            nn.ConvTranspose1d(128, 64, 4, 2, bias=False), nn.InstanceNorm1d(64), nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, 4, 2, bias=False), nn.InstanceNorm1d(32), nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConstantPad1d((1, 1), 0),
            nn.Conv1d(32, 1, 4, padding=1, padding_mode='replicate'),
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(1, 32, 4, 2, 0, bias=False, padding_mode='replicate'), nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2, 0, bias=False, padding_mode='replicate'), nn.InstanceNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 0, bias=False, padding_mode='replicate'), nn.InstanceNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Conv1d(128, 128, 4, 2, bias=False), nn.ReLU(), nn.Dropout(0.5)
        )
        self.fc1 = nn.Linear(101, 100)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_img = self.decoder(latent_i)
        fin_cov = self.final(gen_img)
        fin_cov = fin_cov.view(256, -1)
        y = self.fc1(fin_cov)
        latent_o = self.encoder2(y.reshape(256, 1, -1))
        return y, latent_o

def gan_rec(ppg_clean, noise, sampling_rate, generator, device):
    shifting_step = 5
    len_rec = int(shifting_step * sampling_rate)
    reconstructed_noise = []
    remaining_noise = len(noise)

    while len(reconstructed_noise) < len(noise):
        ppg_clean = preprocessing.scale(ppg_clean)
        y = np.array([ppg_clean for _ in range(256)])
        ppg_test = torch.FloatTensor(y)

        _, rec_latent = generator(ppg_test.reshape(256, 1, -1).to(device))
        rec_test = rec_latent.cpu().detach().numpy()[0]

        rec_resampled = resample(rec_test, int(len(rec_test) * UPSAMPLING_RATE))
        ppg_resampled = resample(ppg_clean, int(len(ppg_clean) * UPSAMPLING_RATE))

        peaks_rec, _ = find_peaks(rec_resampled.flatten(), int(sampling_rate * UPSAMPLING_RATE))
        peaks_ppg, _ = find_peaks(ppg_resampled.flatten(), int(sampling_rate * UPSAMPLING_RATE))

        if len(peaks_ppg) == 0 or len(peaks_rec) == 0:
            downsampled = resample(ppg_clean, len(ppg_clean) + len(rec_test))
        else:
            combined = list(ppg_resampled[:peaks_ppg[-1]]) + list(rec_resampled[peaks_rec[0]:])
            downsampled = resample(combined, len(ppg_clean) + len(rec_test))

        if remaining_noise < len_rec:
            reconstructed_noise += list(downsampled[len(ppg_clean):len(ppg_clean) + remaining_noise])
        else:
            reconstructed_noise += list(downsampled[len(ppg_clean):len(ppg_clean) + len_rec])
            remaining_noise -= len_rec

        ppg_clean = downsampled[len_rec:len(ppg_clean) + len_rec]

    return reconstructed_noise

def reconstruction(sig, clean_indices, noisy_indices, sampling_rate, filter_signal=True, persistent_clean_buffer=None) -> Tuple[np.ndarray, list, list]:
    from torch.serialization import add_safe_globals
    add_safe_globals({'Generator': Generator})

    generator = torch.load(
        os.path.join(MODEL_PATH, GAN_MODEL_FILE_NAME),
        map_location="cpu",
        weights_only=False
    )
    generator.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if persistent_clean_buffer is None:
        persistent_clean_buffer = deque(maxlen=5)

    resampling_flag = False
    if sampling_rate != RECONSTRUCTION_MODEL_SAMPLING_FREQUENCY:
        sig = resample_signal(sig, sampling_rate, RECONSTRUCTION_MODEL_SAMPLING_FREQUENCY)
        resampling_flag = True
        resampling_rate = sampling_rate / RECONSTRUCTION_MODEL_SAMPLING_FREQUENCY
        sampling_rate_original = sampling_rate
        sampling_rate = RECONSTRUCTION_MODEL_SAMPLING_FREQUENCY
        clean_indices = [list(range(int(g[0]/resampling_rate), int(g[-1]/resampling_rate)+1)) for g in clean_indices]
        clean_indices = [i for g in clean_indices for i in g]
        signal_len = len(sig)
        signal_indices = list(range(signal_len))
        noisy_indices = [list(g) for g in mit.consecutive_groups([i for i in signal_indices if i not in clean_indices])]
    else:
        clean_indices = [i for g in clean_indices for i in g]

    if filter_signal:
        sig = bandpass_filter(sig, sampling_rate)

    global_clean = sig[clean_indices]
    global_std = np.std(global_clean)
    global_mean = np.mean(global_clean)
    clip_limit = 1.5 * global_std

    sig_centered = sig - np.mean(sig)
    sig = np.clip(sig_centered, -clip_limit, clip_limit) + global_mean

    sig_scaled = preprocessing.scale(sig)
    max_rec_len = int(MAX_RECONSTRUCTION_LENGTH_SEC * sampling_rate)
    reconstruction_flag = False

    for noise in noisy_indices:
        use_fallback = False
        valid_range = range(noise[0] - max_rec_len, noise[0])

        if len(noise) <= max_rec_len and noise[0] >= max_rec_len:
            if set(valid_range).issubset(clean_indices):
                clean_ref = sig[noise[0] - max_rec_len:noise[0]]
                persistent_clean_buffer.append(clean_ref)
            elif len(persistent_clean_buffer) > 0:
                clean_ref = persistent_clean_buffer[-1]
                use_fallback = True
            else:
                continue
        else:
            continue

        extend = int(0.5 * sampling_rate)
        ext_start = max(0, noise[0] - extend)
        ext_end = min(len(sig), noise[-1] + extend + 1)
        extended_noise = list(range(ext_start, ext_end))

        noisy_ref = sig[extended_noise]

        clean_std = np.std(clean_ref)
        clean_mean = np.mean(clean_ref)
        local_clip = 1.5 * clean_std

        noisy_centered = noisy_ref - np.mean(noisy_ref)
        clipped = np.clip(noisy_centered, -local_clip, local_clip)
        sig[extended_noise] = clipped + clean_mean

        rec_noise = gan_rec(clean_ref, noise, sampling_rate, generator, device)
        sig_scaled[noise[0]:noise[-1]+1] = resample(rec_noise, len(noise))
        reconstruction_flag = True

    if reconstruction_flag:
        ppg_descaled = (sig_scaled * np.std(sig)) + np.mean(sig)
    else:
        ppg_descaled = sig

    if resampling_flag:
        ppg_descaled = resample_signal(ppg_descaled, sampling_rate, sampling_rate_original)
        sampling_rate = sampling_rate_original

    ppg_descaled = savgol_filter(ppg_descaled, window_length=31, polyorder=3)

    clean_indices, noisy_indices = sqa(ppg_descaled, sampling_rate, filter_signal=False)
    return ppg_descaled, clean_indices, noisy_indices
