import sys
import json
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from collections import deque

# === Constants ===
FS = 50.0
CUTOFF = 5.0
WAVELET = 'sym4'
LEVEL = 4

# === Smoothing buffers ===
spo2_buffer = deque(maxlen=5)
last_spo2 = None

def wavelet_denoise(signal):
    coeffs = pywt.wavedec(signal, WAVELET, level=LEVEL)
    for i in [3, 4]:
        coeffs[i] = np.zeros_like(coeffs[i])
    return pywt.waverec(coeffs, WAVELET)[:len(signal)]

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    norm = cutoff / nyq
    b, a = butter(order, norm, btype='low')
    return b, a

def apply_lowpass_filter(signal, cutoff=5, fs=FS, order=2):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, signal)

def apply_savgol_filter(signal, window_length=7, polyorder=2):
    if len(signal) < window_length:
        return signal
    return savgol_filter(signal, window_length, polyorder)

def detect_peaks(signal):
    peaks, _ = find_peaks(signal, distance=int(FS * 0.4), height=0.1)
    return peaks

def detect_valleys(signal):
    inverted = -np.array(signal)
    valleys, _ = find_peaks(inverted, distance=int(FS * 0.4), prominence=0.5)
    return valleys

def calculate_intersection(peak_x, valley_x1, valley_y1, valley_x2, valley_y2):
    if valley_x1 == valley_x2:
        return peak_x, (valley_y1 + valley_y2) / 2
    slope = (valley_y2 - valley_y1) / (valley_x2 - valley_x1)
    intercept = valley_y1 - slope * valley_x1
    return peak_x, slope * peak_x + intercept

def process_channel(peaks, valleys, signal):
    ac_list, dc_list = [], []
    for peak in peaks:
        left = [v for v in valleys if v < peak]
        right = [v for v in valleys if v > peak]
        if not left or not right:
            continue
        v1 = left[-1]
        v2 = right[0]
        if v1 < 0 or v2 >= len(signal) or peak >= len(signal):
            continue
        x, y = calculate_intersection(peak, v1, signal[v1], v2, signal[v2])
        ac = signal[peak] - y
        dc = y
        if dc > 0 and ac > 0:
            ac_list.append(ac)
            dc_list.append(dc)
    return np.array(ac_list), np.array(dc_list)

def compute_spo2(red_signal, ir_signal, timestamps):
    global last_spo2

    if len(red_signal) < 20 or len(ir_signal) < 20 or len(timestamps) < 20:
        raise ValueError("Insufficient data for SpO₂ analysis.")
    if len(red_signal) != len(ir_signal) or len(red_signal) != len(timestamps):
        raise ValueError("Signal and timestamp lengths mismatch.")
    if np.any(np.isnan(red_signal)) or np.any(np.isnan(ir_signal)) or np.any(np.isnan(timestamps)):
        raise ValueError("Input contains NaN values.")
    if np.ptp(red_signal) < 100 or np.ptp(ir_signal) < 100:
        raise ValueError("Signals too flat to analyze.")

    red = apply_savgol_filter(apply_lowpass_filter(wavelet_denoise(red_signal)))
    ir = apply_savgol_filter(apply_lowpass_filter(wavelet_denoise(ir_signal)))

    red_peaks = detect_peaks(red)
    ir_peaks = detect_peaks(ir)
    red_valleys = detect_valleys(red)
    ir_valleys = detect_valleys(ir)

    ac_red, dc_red = process_channel(red_peaks, red_valleys, red)
    ac_ir, dc_ir = process_channel(ir_peaks, ir_valleys, ir)

    n = min(len(ac_red), len(dc_red), len(ac_ir), len(dc_ir))
    if n < 3:
        raise ValueError("Too few valid AC/DC components for SpO₂.")

    ac_red, dc_red = ac_red[:n], dc_red[:n]
    ac_ir, dc_ir = ac_ir[:n], dc_ir[:n]

    if np.mean(ac_red) < 5 or np.mean(ac_ir) < 5:
        raise ValueError("Weak AC signal, skipping")

    try:
        R = (ac_red / dc_red) / (ac_ir / dc_ir)
    except Exception as e:
        raise ValueError(f"Ratio computation failed: {str(e)}")

    R = R[(R > 0.4) & (R < 1.0)]
    if len(R) == 0:
        raise ValueError("Invalid R values after filtering.")

    spo2 = 119 - 25 * R
    spo2 = spo2[(spo2 >= 85) & (spo2 <= 100)]
    if len(spo2) == 0:
        raise ValueError("SpO₂ values out of range after computation.")

    raw_avg = np.mean(spo2)

    # Clamp jump from last_spo2
    if last_spo2 is not None:
        diff = raw_avg - last_spo2
        if abs(diff) > 2.5:
            raw_avg = last_spo2 + (2.5 if diff > 0 else -2.5)

    last_spo2 = raw_avg

    # Smooth over buffer
    spo2_buffer.append(raw_avg)
    smoothed = np.mean(spo2_buffer)

    clamped = min(smoothed, 100)
    return round(clamped, 2)

# === MAIN ===
if __name__ == "__main__":
    try:
        input_data = json.load(sys.stdin)
        red = np.array(input_data["red"], dtype=float)
        ir = np.array(input_data["ir"], dtype=float)
        timestamps = np.array(input_data["timestamps"], dtype=float)

        spo2 = compute_spo2(red, ir, timestamps)
        print(json.dumps({"spo2": spo2}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
