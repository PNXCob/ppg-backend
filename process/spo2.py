import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
import sys
import json

FS = 50.0  # Hz

# --- Filters ---
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def apply_lowpass_filter(signal, cutoff=5, fs=FS, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, signal)

def wavelet_denoise(signal, wavelet='sym4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[2] = np.zeros_like(coeffs[2])
    coeffs[3] = np.zeros_like(coeffs[3])
    coeffs[4] = np.zeros_like(coeffs[4])
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

def apply_savgol_filter(signal, window_length=7, polyorder=2):
    if len(signal) < window_length:
        return signal
    return savgol_filter(signal, window_length, polyorder)

# --- Peak & Valley Detection ---
def detect_peaks(signal, fs):
    return find_peaks(signal, prominence=10, distance=int(fs * 1))[0]

def detect_valleys(signal, fs):
    inverted_signal = -np.array(signal)
    return find_peaks(inverted_signal, prominence=10, distance=int(fs * 1))[0]

# --- Calculate SpO2 ---
def calculate_spo2(red_signal, ir_signal, fs):
    peaks_red = detect_peaks(red_signal, fs)
    valleys_red = detect_valleys(red_signal, fs)
    peaks_ir = detect_peaks(ir_signal, fs)
    valleys_ir = detect_valleys(ir_signal, fs)

    if len(peaks_red) < 2 or len(valleys_red) < 2 or len(peaks_ir) < 2 or len(valleys_ir) < 2:
        return None

    valid_ac_red, valid_dc_red = [], []
    valid_ac_ir, valid_dc_ir = [], []

    for peak, valley1, valley2 in zip(peaks_red, valleys_red[:-1], valleys_red[1:]):
        ac = red_signal[peak] - ((red_signal[valley1] + red_signal[valley2]) / 2)
        dc = (red_signal[valley1] + red_signal[valley2]) / 2
        valid_ac_red.append(ac)
        valid_dc_red.append(dc)

    for peak, valley1, valley2 in zip(peaks_ir, valleys_ir[:-1], valleys_ir[1:]):
        ac = ir_signal[peak] - ((ir_signal[valley1] + ir_signal[valley2]) / 2)
        dc = (ir_signal[valley1] + ir_signal[valley2]) / 2
        valid_ac_ir.append(ac)
        valid_dc_ir.append(dc)

    if len(valid_ac_red) == 0 or len(valid_ac_ir) == 0:
        return None

    R = (np.array(valid_ac_red) / np.array(valid_dc_red)) / (np.array(valid_ac_ir) / np.array(valid_dc_ir))
    R = np.clip(R, 0.4, 1.0)

    spo2 = 110 - 25 * np.mean(R)
    return float(f"{spo2:.2f}")

# --- Main Processing ---
def process_ppg(red, ir):
    red = np.array(red)
    ir = np.array(ir)

    red = apply_lowpass_filter(wavelet_denoise(red))
    ir = apply_lowpass_filter(wavelet_denoise(ir))

    red = apply_savgol_filter(red)
    ir = apply_savgol_filter(ir)

    spo2 = calculate_spo2(red, ir, FS)

    return {
        "spo2": spo2,
        "red_processed": red.tolist(),
        "ir_processed": ir.tolist()
    }

# --- Execution for server.js ---
if __name__ == "__main__":
    red_signal = json.loads(sys.argv[1])
    ir_signal = json.loads(sys.argv[2])

    result = process_ppg(red_signal, ir_signal)
    print(json.dumps(result))
