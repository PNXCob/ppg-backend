import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, medfilt
from scipy.interpolate import interp1d
import sys
import json

FS = 50.0  # Sampling Frequency

# --- Bandpass Filter ---
def apply_bandpass_filter(signal, fs, lowcut=0.5, highcut=3, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# --- Wavelet Denoising ---
def wavelet_denoising(signal, wavelet='sym4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[2] = np.zeros_like(coeffs[2])   # Remove D3
    coeffs[3] = np.zeros_like(coeffs[3])   # Remove D2
    coeffs[4] = np.zeros_like(coeffs[4])   # Remove D1
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

# --- Smoothing Filters ---
def apply_savgol_filter(signal, window_length=19, polyorder=2):
    if len(signal) < window_length:
        return signal
    return savgol_filter(signal, window_length, polyorder)

def apply_moving_average(signal, window_size=23):
    if len(signal) < window_size:
        return signal
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# --- Enhance PPG Pulse ---
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

# --- Peak Detection ---
def detect_peaks(signal, fs, distance=10, height=0.5, rr_threshold_ratio=0.3):
    peaks, _ = find_peaks(signal, distance=distance, height=height)
    if len(peaks) < 2:
        return peaks

    rr_intervals = np.diff(peaks)
    rr_intervals_sec = rr_intervals / fs
    mean_rr_sec = np.mean(rr_intervals_sec)
    min_rr_samples = rr_threshold_ratio * mean_rr_sec * fs

    refined_peaks = [peaks[0]]
    for i in range(1, len(peaks)):
        if (peaks[i] - refined_peaks[-1]) >= min_rr_samples:
            refined_peaks.append(peaks[i])

    return np.array(refined_peaks)

# --- Calculate BPM ---
def calculate_heart_rate(peaks, window_duration=10):
    bpm = len(peaks) * 6  # For 10s window
    return bpm

# --- Main Function ---
def process_ppg(ppg_raw, timestamps):
    timestamps = np.array(timestamps)
    ppg_raw = np.array(ppg_raw)

    ppg_upsampled, _ = upsample_signal(timestamps, ppg_raw, target_fs=FS)
    denoised_signal = wavelet_denoising(ppg_upsampled)
    filtered_signal = apply_bandpass_filter(denoised_signal, FS, lowcut=0.5, highcut=3)
    inverted_signal = -1 * filtered_signal
    smoothed_signal = apply_savgol_filter(inverted_signal)
    smoothed_signal = apply_moving_average(smoothed_signal)
    enhanced_signal = enhance_pulse(smoothed_signal)

    peaks = detect_peaks(enhanced_signal, FS, distance=int(FS * 0.05), height=0.1)
    bpm = calculate_heart_rate(peaks)

    return {
        "bpm": bpm,
        "processed_signal": enhanced_signal.tolist()
    }

# --- For server.js call ---
def upsample_signal(timestamps, signal, target_fs):
    duration = timestamps[-1] - timestamps[0]
    num_samples = int(duration * target_fs)
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_samples)
    interpolator = interp1d(timestamps, signal, kind='linear', fill_value='extrapolate')
    return interpolator(new_timestamps), new_timestamps

if __name__ == "__main__":
    ppg_raw = json.loads(sys.argv[1])
    timestamps = json.loads(sys.argv[2])

    result = process_ppg(ppg_raw, timestamps)
    print(json.dumps(result))
