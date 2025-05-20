import sys
import json
import numpy as np
import pywt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

# --- Settings ---
FS = 50
BUFFER_SECONDS = 15
BUFFER_SIZE = FS * BUFFER_SECONDS

def compute_precise_rr(signal, fs):
    coeffs = pywt.wavedec(signal, 'sym4', level=5)
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])
    baseline = pywt.waverec(coeffs, 'sym4')[:len(signal)]
    smoothed = savgol_filter(baseline, 31, 3)

    peaks, _ = find_peaks(smoothed, distance=fs * 2.5, prominence=10)

    if len(peaks) < 2:
        raise ValueError("Too few peaks to calculate RR.")

    # Use average interval between peaks
    intervals = np.diff(peaks) / fs  # in seconds
    avg_interval_sec = np.mean(intervals)
    rr_bpm = 60.0 / avg_interval_sec

    return int(round(rr_bpm))  # <- ✅ integer only

# --- Main Logic ---
try:
    input_json = sys.stdin.read()
    data = json.loads(input_json)
    green = np.array(data["green"])
    timestamps = np.array(data["timestamps"])

    if len(green) < BUFFER_SIZE:
        raise ValueError("Insufficient data: need 750 samples")

    t_shifted = timestamps - timestamps[0]
    t_uniform = np.linspace(t_shifted[0], t_shifted[-1], BUFFER_SIZE)
    uniform_signal = interp1d(t_shifted, green, fill_value="extrapolate")(t_uniform)

    rr_bpm = compute_precise_rr(uniform_signal, FS)

    if rr_bpm < 6 or rr_bpm > 30:
        raise ValueError(f"Unrealistic RR: {rr_bpm}")

    print(json.dumps({ "rr": rr_bpm }))
except Exception as e:
    print(json.dumps({ "error": str(e) }))
