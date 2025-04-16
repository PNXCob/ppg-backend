import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, decimate
from scipy.interpolate import interp1d
import sys
import json

# --- Bandpass Filter ---
def bandpass_filter(signal, fs, lowcut=0.1, highcut=30.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# --- Upsampling ---
def upsample_signal(timestamps, signal, target_fs):
    duration = timestamps[-1] - timestamps[0]
    num_samples = int(duration * target_fs)
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_samples)
    interpolator = interp1d(timestamps, signal, kind='linear', fill_value='extrapolate')
    return interpolator(new_timestamps), new_timestamps

# --- Wavelet Denoising ---
def wavelet_denoising(signal, wavelet='db8', level=13):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # A13
    coeffs[1] = np.zeros_like(coeffs[1])  # D13
    coeffs[2] = np.zeros_like(coeffs[2])  # D12
    coeffs[13] = np.zeros_like(coeffs[13])  # D1
    return pywt.waverec(coeffs, wavelet=wavelet)

# --- Estimate BP from peaks/troughs only ---
def estimate_bp(signal, fs=125):
    peaks, _ = find_peaks(signal, distance=50, prominence=10)
    sbp_values = signal[peaks]

    troughs, _ = find_peaks(-signal, distance=50, prominence=10)
    dbp_values = signal[troughs]

    sbp = np.mean(sbp_values)
    dbp = np.mean(dbp_values)

    return float(f"{sbp:.1f}"), float(f"{dbp:.1f}")

# --- Process Raw PPG and Estimate BP ---
def process_ppg(ppg_raw, timestamps):
    ppg_upsampled, _ = upsample_signal(timestamps, ppg_raw, target_fs=1000)
    ppg_denoised = wavelet_denoising(ppg_upsampled)
    ppg_downsampled = decimate(ppg_denoised, q=8)
    ppg_filtered = bandpass_filter(ppg_downsampled, fs=1000)
    ppg_inverted = -1 * ppg_filtered
    ppg_norm = (ppg_inverted - np.min(ppg_inverted)) / (np.max(ppg_inverted) - np.min(ppg_inverted))
    ppg_smoothed = savgol_filter(ppg_norm, window_length=17, polyorder=2)

    sbp, dbp = estimate_bp(ppg_smoothed)

    return {
        "sbp": sbp,
        "dbp": dbp,
        "processed_signal": ppg_smoothed.tolist()
    }

# === Entry point for server.js ===
if __name__ == "__main__":
    ppg_raw = json.loads(sys.argv[1])
    timestamps = json.loads(sys.argv[2])

    result = process_ppg(ppg_raw, timestamps)
    print(json.dumps(result))
