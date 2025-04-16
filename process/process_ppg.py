import numpy as np
from utils import (
    bandpass_filter,
    wavelet_denoising,
    upsample_signal,
    enhance_pulse,
    estimate_bp,
    detect_peaks,
    calculate_heart_rate,
    compute_rr_intervals,
    smoother_rr_signal,
    extract_dominant_frequency
)

def preprocess_ppg(ppg_raw, timestamps, fs=50, target_fs=1000):
    # Upsample
    ppg_upsampled, _ = upsample_signal(timestamps, ppg_raw, target_fs)

    # Wavelet Denoising
    ppg_denoised = wavelet_denoising(ppg_upsampled)

    # Bandpass Filtering
    ppg_filtered = bandpass_filter(ppg_denoised, target_fs)

    # Invert & Normalize
    ppg_inverted = -1 * ppg_filtered
    ppg_norm = (ppg_inverted - np.min(ppg_inverted)) / (np.max(ppg_inverted) - np.min(ppg_inverted))

    return ppg_norm.tolist()


def process_bp(ppg_raw, timestamps):
    processed_ppg = preprocess_ppg(ppg_raw, timestamps)

    # Dummy ABP Simulation
    sbp, dbp = estimate_bp(np.array(processed_ppg))

    return {
        "sbp": sbp,
        "dbp": dbp
    }


def process_hr(ppg_raw, timestamps, fs=50):
    processed_ppg = preprocess_ppg(ppg_raw, timestamps)

    # Detect Peaks
    peaks = detect_peaks(np.array(processed_ppg), fs=fs, distance=int(fs*0.2), height=0.2)

    bpm = calculate_heart_rate(peaks, window_duration=10)

    return {
        "bpm": bpm
    }


def process_rr(peaks, fs=50):
    rr_intervals = compute_rr_intervals(peaks, fs)
    x_interp, rr_interp = smoother_rr_signal(rr_intervals, fs_interp=1000)

    freq = extract_dominant_frequency(rr_interp, fs=1000)
    rr = freq * 60

    return {
        "resp_rate": rr
    }
