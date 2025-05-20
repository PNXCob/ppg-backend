import sys
import json
import numpy as np
from scipy.signal import find_peaks, medfilt
from sqa_advanced_utils import wavelet_denoising, bandpass_filter
from reconstruction_utils import reconstruction

FS = 50.0  # Sampling frequency

def enhance_pulse(signal, window_size=25):
    baseline = medfilt(signal, kernel_size=window_size)
    signal = signal - baseline
    windowed_max = np.maximum.accumulate(signal)
    windowed_max[windowed_max == 0] = 1
    signal = signal / windowed_max
    signal = np.sign(signal) * np.abs(signal) ** 0.5
    signal = signal - np.min(signal)
    if np.max(signal) > 0:
        signal = signal / np.max(signal)
    return signal

def detect_peaks(signal, fs, distance=10, height=0.1, rr_threshold_ratio=0.3):
    peaks, _ = find_peaks(signal, distance=distance, height=height)
    if len(peaks) < 2:
        return peaks
    rr_intervals = np.diff(peaks)
    rr_intervals_sec = rr_intervals / fs
    mean_rr_sec = np.mean(rr_intervals_sec)
    min_rr_sec = rr_threshold_ratio * mean_rr_sec
    min_rr_samples = min_rr_sec * fs
    refined_peaks = [peaks[0]]
    for i in range(1, len(peaks)):
        if (peaks[i] - refined_peaks[-1]) >= min_rr_samples:
            refined_peaks.append(peaks[i])
    return np.array(refined_peaks)

def process_ppg(ppg, time, fs):
    if len(ppg) < 2 or len(time) < 2:
        raise ValueError("Insufficient data points.")
    if np.any(np.isnan(ppg)) or np.ptp(ppg) < 100:
        raise ValueError("Invalid or flat PPG.")

    denoised = wavelet_denoising(ppg)
    filtered = bandpass_filter(denoised, fs)

    # Only one enhancement pass
    current_signal = enhance_pulse(filtered)

    # Force at least 0.6s between peaks = ~100 BPM max
    distance = int(fs * 0.6)
    peaks = detect_peaks(current_signal, fs=fs, distance=distance, height=0.1)

    if len(peaks) < 2:
        raise ValueError("Too few peaks detected.")

    duration_sec = time[-1] - time[0]
    bpm = (len(peaks) / duration_sec) * 60

    # Final realistic clamp for seated test
    if bpm < 20 or bpm > 150:
        raise ValueError(f"Unrealistic BPM detected: {bpm}")

    return int(round(bpm))

if __name__ == "__main__":
    try:
        data = json.loads(sys.stdin.read())
        ppg = np.array(data["ppg"], dtype=float)
        timestamps = np.array(data["timestamps"], dtype=float)

        if len(ppg) != len(timestamps):
            raise ValueError("Length mismatch.")
        fs = 1.0 / np.mean(np.diff(timestamps))
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("Invalid sampling rate.")

        bpm = process_ppg(ppg, timestamps, fs)
        print(json.dumps({"bpm": bpm}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
