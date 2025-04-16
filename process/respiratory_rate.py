import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, medfilt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d, UnivariateSpline
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
    coeffs[2] = np.zeros_like(coeffs[2])
    coeffs[3] = np.zeros_like(coeffs[3])
    coeffs[4] = np.zeros_like(coeffs[4])
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
def detect_peaks(signal, fs, distance=10, height=0.5):
    peaks, _ = find_peaks(signal, distance=distance, height=height)
    return peaks

# --- RR Interval & Breathing Rate ---
def compute_rr_intervals(peaks, fs):
    return np.diff(peaks) / fs

def smoother_rr_signal(rr_intervals, fs_interp=1000, smoothing_factor=0.0001):
    rr_cumsum = np.cumsum(rr_intervals)
    x_new = np.linspace(0, rr_cumsum[-1], int(rr_cumsum[-1] * fs_interp))
    spline = UnivariateSpline(rr_cumsum, rr_intervals, s=smoothing_factor)
    rr_interpolated_signal = spline(x_new)
    return x_new, rr_interpolated_signal

def extract_dominant_frequency(rr_signal, fs, low_freq=0.1, high_freq=0.4):
    rr_fft = fft(rr_signal)
    freqs = fftfreq(len(rr_signal), 1/fs)
    rr_fft = np.abs(rr_fft[:len(rr_fft)//2])
    freqs = freqs[:len(freqs)//2]
    valid = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    freqs_valid = freqs[valid]
    rr_fft_valid = rr_fft[valid]
    dominant_freq = freqs_valid[np.argmax(rr_fft_valid)]
    return dominant_freq

# --- Process PPG Data ---
def process_ppg(ppg_raw, timestamps):
    timestamps = np.array(timestamps)
    ppg_raw = np.array(ppg_raw)

    ppg_upsampled, _ = upsample_signal(timestamps, ppg_raw, target_fs=FS)
    denoised = wavelet_denoising(ppg_upsampled)
    filtered = apply_bandpass_filter(denoised, FS, lowcut=0.5, highcut=3)
    smoothed = apply_savgol_filter(filtered)
    smoothed = apply_moving_average(smoothed)
    enhanced = enhance_pulse(smoothed)

    peaks = detect_peaks(enhanced, FS, distance=int(FS * 0.05), height=0.1)
    if len(peaks) < 2:
        return {"breathing_rate": 0, "processed_signal": enhanced.tolist()}

    rr_intervals = compute_rr_intervals(peaks, FS)
    x_interp, rr_signal_interp = smoother_rr_signal(rr_intervals, fs_interp=1000)
    dominant_freq = extract_dominant_frequency(rr_signal_interp, fs=1000)
    breathing_rate = dominant_freq * 60  # breaths per minute

    return {
        "breathing_rate": float(f"{breathing_rate:.2f}"),
        "processed_signal": enhanced.tolist()
    }

# --- Upsample ---
def upsample_signal(timestamps, signal, target_fs):
    duration = timestamps[-1] - timestamps[0]
    num_samples = int(duration * target_fs)
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_samples)
    interpolator = interp1d(timestamps, signal, kind='linear', fill_value='extrapolate')
    return interpolator(new_timestamps), new_timestamps

# --- Server Call ---
if __name__ == "__main__":
    ppg_raw = json.loads(sys.argv[1])
    timestamps = json.loads(sys.argv[2])
    result = process_ppg(ppg_raw, timestamps)
    print(json.dumps(result))
