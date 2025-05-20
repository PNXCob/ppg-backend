import sys
import json
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, decimate
from scipy.interpolate import interp1d
import os

# === Suppress TensorFlow AVX log spam ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === Fix import path ===
sys.path.append(os.path.join(os.path.dirname(__file__)))
from models import UNetDS64, MultiResUNet1D

# Constants
FS_ORIG = 125
FS_UP = 1000

# Load models with dynamic path
model_dir = os.path.join(os.path.dirname(__file__), "models")
model1 = UNetDS64(1024)
model1.load_weights(os.path.join(model_dir, "ApproximateNetwork.h5"))

model2 = MultiResUNet1D(1024)
model2.load_weights(os.path.join(model_dir, "RefinementNetwork.h5"))

# --- Bandpass Filter ---
def bandpass_filter(signal, fs, lowcut=0.1, highcut=30.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# --- Upsample Signal ---
def upsample_signal(timestamps, signal, target_fs):
    duration = timestamps[-1] - timestamps[0]
    num_samples = int(duration * target_fs)
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_samples)
    interpolator = interp1d(timestamps, signal, kind='linear', fill_value='extrapolate')
    return interpolator(new_timestamps), new_timestamps

# --- Wavelet Denoising ---
def wavelet_denoising(signal, wavelet='db8', level=11):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    # Zero out approximation and most detailed levels
    for i in [0, 1, 2, -1]:
        coeffs[i] = np.zeros_like(coeffs[i])
    return pywt.waverec(coeffs, wavelet=wavelet)

# --- Estimate BP ---
def estimate_bp(abp_signal, fs=125):
    peaks, _ = find_peaks(abp_signal, distance=50, prominence=10)
    troughs, _ = find_peaks(-abp_signal, distance=50, prominence=15)
    sbp = np.mean(abp_signal[peaks]) if len(peaks) else 0
    dbp = np.median(abp_signal[troughs]) if len(troughs) else 0
    return sbp, dbp

# --- Main BP Processor ---
def process_bp(ppg_green, timestamps):
    if len(ppg_green) < 100 or len(ppg_green) != len(timestamps):
        raise ValueError("Insufficient or mismatched PPG and timestamps")

    timestamps = np.array(timestamps) - timestamps[0]
    upsampled, _ = upsample_signal(timestamps, ppg_green, FS_UP)
    denoised = wavelet_denoising(upsampled)
    downsampled = decimate(denoised, q=8)
    filtered = bandpass_filter(downsampled, fs=FS_ORIG)

    inverted = -1 * filtered
    range_val = np.max(inverted) - np.min(inverted)
    if range_val == 0:
        raise ValueError("Flat signal: cannot normalize")
    norm = (inverted - np.min(inverted)) / range_val

    padded = norm if len(norm) == 1024 else np.pad(norm, (0, max(0, 1024 - len(norm))))[:1024]
    smoothed = savgol_filter(padded, window_length=17, polyorder=2)
    model_input = np.expand_dims(smoothed.reshape(1024, 1).astype(np.float32), axis=0)

    approx_output = model1.predict(model_input, verbose=0)[0]
    refined_output = model2.predict(approx_output, verbose=0)

    abp_mmHg = refined_output[0, :, 0] * (190 - 60) + 60
    sbp, dbp = estimate_bp(abp_mmHg, fs=FS_ORIG)

    return {
        "sbp": round(float(sbp), 2),
        "dbp": round(float(dbp), 2),
        "bp_waveform": abp_mmHg.tolist()
    }

# === CLI Mode ===
if __name__ == "__main__":
    try:
        data = json.load(sys.stdin)
        green = np.array(data["green"], dtype=float)
        timestamps = np.array(data["timestamps"], dtype=float)
        result = process_bp(green, timestamps)
        print(json.dumps(result))  # ✅ Print only clean JSON
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
