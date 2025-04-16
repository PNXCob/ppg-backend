import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, decimate, medfilt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.fft import fft, fftfreq

def bandpass_filter(signal, fs, lowcut=0.1, highcut=30.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def wavelet_denoising(signal, wavelet='db8', level=13):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])
    coeffs[1] = np.zeros_like(coeffs[1])
    coeffs[2] = np.zeros_like(coeffs[2])
    coeffs[13] = np.zeros_like(coeffs[13])
    return pywt.waverec(coeffs, wavelet=wavelet)

def upsample_signal(timestamps, signal, target_fs):
    duration = timestamps[-1] - timestamps[0]
    num_samples = int(duration * target_fs)
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_samples)
    interpolator = interp1d(timestamps, signal, kind='linear', fill_value='extrapolate')
    return interpolator(new_timestamps), new_timestamps

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

def estimate_bp(abp_signal):
    peaks, _ = find_peaks(abp_signal, distance=50, prominence=10)
    troughs, _ = find_peaks(-abp_signal, distance=50, prominence=10)
    sbp = np.mean(abp_signal[peaks])
    dbp = np.mean(abp_signal[troughs])
    return float(f"{sbp:.1f}"), float(f"{dbp:.1f}")

def detect_peaks(signal, fs, distance=10, height=0.5):
    peaks, _ = find_peaks(signal, distance=distance, height=height)
    return peaks

def calculate_heart_rate(peaks, window_duration=10):
    return len(peaks) * (60 / window_duration)

def compute_rr_intervals(peaks, fs):
    return np.diff(peaks) / fs

def smoother_rr_signal(rr_intervals, fs_interp=1000, smoothing_factor=0.5):
    rr_cumsum = np.cumsum(rr_intervals)
    x_new = np.linspace(0, rr_cumsum[-1], int(rr_cumsum[-1] * fs_interp))
    spline = UnivariateSpline(rr_cumsum, rr_intervals, s=smoothing_factor)
    rr_interp = spline(x_new)
    return x_new, rr_interp

def extract_dominant_frequency(rr_signal, fs, low_freq=0.1, high_freq=0.4):
    rr_fft = fft(rr_signal)
    freqs = fftfreq(len(rr_signal), 1/fs)
    rr_fft = np.abs(rr_fft[:len(rr_fft)//2])
    freqs = freqs[:len(freqs)//2]
    valid_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    return freqs[valid_idx][np.argmax(rr_fft[valid_idx])]
