import numpy as np
import pywt
import joblib
import pickle
import os
import more_itertools as mit
from scipy import signal, stats

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")
SCALER_FILE_NAME = "Train_data_scaler.save"
SQA_MODEL_FILE_NAME = "OneClassSVM_model.sav"
SQA_MODEL_SAMPLING_FREQUENCY = 20
SHIFTING_SIZE = 2

def normalize_data(sig):
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig))

def resample_signal(sig, fs_origin, fs_target):
    duration = len(sig) / fs_origin
    target_length = int(duration * fs_target)
    return signal.resample(sig, target_length)

def wavelet_denoising(sig, wavelet='sym4', level=4):
    sig = np.asarray(sig).flatten()
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    for i in range(2, 4):
        coeffs[i] = np.zeros_like(coeffs[i])
    return pywt.waverec(coeffs, wavelet)[:len(sig)]

def bandpass_filter(sig, fs, lowcut=0.5, highcut=3, order=2):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, sig)

def find_peaks(ppg, sampling_rate, return_sig=False):
    distance = int(sampling_rate * 0.4)
    peaks, _ = signal.find_peaks(ppg, distance=distance)
    return (peaks, ppg) if return_sig else peaks

def segmentation(sig, sig_indices, sampling_rate, method='shifting', segment_size=4.5, shift_size=1):
    segment_len = int(segment_size * sampling_rate)
    shift_len = int(shift_size * sampling_rate)
    if method == 'standard':
        segments_sig = [sig[i:i + segment_len] for i in range(0, len(sig), segment_len) if i + segment_len <= len(sig)]
        segments_idx = [sig_indices[i:i + segment_len] for i in range(0, len(sig), segment_len) if i + segment_len <= len(sig)]
    else:
        segments_sig = [sig[i:i + segment_len] for i in range(0, len(sig) - segment_len + 1, shift_len)]
        segments_idx = [sig_indices[i:i + segment_len] for i in range(0, len(sig) - segment_len + 1, shift_len)]
    return segments_sig, segments_idx

def heart_cycle_detection(ppg, sampling_rate):
    ppg_norm = normalize_data(ppg)
    ppg_up = signal.resample(ppg_norm, len(ppg_norm) * 2)
    sampling_rate *= 2
    peaks, _ = find_peaks(ppg_up, sampling_rate, return_sig=True)
    hc = []
    if len(peaks) < 2:
        return hc
    beat_bound = round((len(ppg_up) / len(peaks)) / 2)
    for i in range(1, len(peaks) - 1):
        start = peaks[i] - beat_bound
        end = peaks[i] + beat_bound
        if start >= 0 and end < len(ppg_up):
            beat = ppg_up[start:end]
            if len(beat) >= beat_bound * 2:
                hc.append(beat)
    return hc

def energy_hc(hc):
    energy = [np.sum(beat ** 2) for beat in hc]
    return max(energy) - min(energy) if energy else 0

def template_matching_features(hc):
    hc = np.array([beat for beat in hc if len(beat) > 0])
    template = np.mean(hc, axis=0)
    distances = [np.linalg.norm(template - beat) for beat in hc]
    corrs = [np.corrcoef(template, beat)[0, 1] for beat in hc]
    return np.mean(distances), np.mean(corrs)

def feature_extraction(ppg, sampling_rate):
    iqr_rate = stats.iqr(ppg, interpolation='midpoint')
    _, pxx_den = signal.periodogram(ppg, sampling_rate)
    std_psd = np.std(pxx_den)
    hc = heart_cycle_detection(ppg, sampling_rate)
    if hc:
        var_energy = energy_hc(hc)
        tm_ave_eu, tm_ave_corr = template_matching_features(hc)
    else:
        var_energy, tm_ave_eu, tm_ave_corr = np.nan, np.nan, np.nan
    return [iqr_rate, std_psd, var_energy, tm_ave_eu, tm_ave_corr]

def sqa(sig, sampling_rate, filter_signal=True):
    signal_indices = list(range(len(sig)))
    scaler = joblib.load(os.path.join(MODEL_PATH, SCALER_FILE_NAME))
    model = pickle.load(open(os.path.join(MODEL_PATH, SQA_MODEL_FILE_NAME), 'rb'))

    if sampling_rate != SQA_MODEL_SAMPLING_FREQUENCY:
        sig = resample_signal(sig, sampling_rate, SQA_MODEL_SAMPLING_FREQUENCY)
        resampling_flag = True
        resampling_rate = sampling_rate / SQA_MODEL_SAMPLING_FREQUENCY
        sampling_rate = SQA_MODEL_SAMPLING_FREQUENCY
    else:
        resampling_flag = False

        sig = wavelet_denoising(sig)

    if filter_signal:
        sig = bandpass_filter(sig, sampling_rate)

    sig_indices = np.arange(len(sig))
    segments, segments_idx = segmentation(sig, sig_indices, sampling_rate)

    clean_idx_all, noisy_idx_all = [], []
    for idx, segment in enumerate(segments):
        features = feature_extraction(segment, sampling_rate)
        if np.isnan(features).any():
            pred = 1
        else:
            pred = model.predict(scaler.transform([features]))
        if pred == 0:
            clean_idx_all.append(segments_idx[idx])
        else:
            noisy_idx_all.append(segments_idx[idx])

    clean_flat = sorted(set([item for sub in clean_idx_all for item in sub]))
    if resampling_flag:
        clean_flat = [int(i * resampling_rate) for i in clean_flat]
    noisy_flat = [i for i in signal_indices if i not in clean_flat]

    clean_idx = [list(group) for group in mit.consecutive_groups(clean_flat)]
    noisy_idx = [list(group) for group in mit.consecutive_groups(noisy_flat)]
    noisy_idx = [group for group in noisy_idx if len(group) > SHIFTING_SIZE]

    return clean_idx, noisy_idx

def is_clean_fft(segment, fs, band=(0.5, 3.0), threshold=0.7):
    segment = segment - np.mean(segment)
    if np.std(segment) > 0:
        segment = segment / np.std(segment)

    n = len(segment)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_vals = np.abs(np.fft.rfft(segment)) ** 2

    total_power = np.sum(fft_vals)
    band_power = np.sum(fft_vals[(freqs >= band[0]) & (freqs <= band[1])])
    if total_power == 0:
        return False

    return (band_power / total_power) >= threshold

def refine_noisy_segments_fft(ppg_signal, noisy_idx, fs, clean_ref=None, plot_debug=False, amp_thresh_factor=1.5):
    refined_clean, refined_noisy = [], []

    if clean_ref is None or len(clean_ref) == 0:
        clean_ref = []
        use_amp_check = False
    else:
        clean_amplitudes = [np.ptp(ppg_signal[seg[0]:seg[-1]+1]) for seg in clean_ref]
        amp_mean = np.mean(clean_amplitudes)
        amp_std = np.std(clean_amplitudes)
        amp_thresh = amp_mean + amp_thresh_factor * amp_std
        use_amp_check = True

    for i, seg in enumerate(noisy_idx):
        segment = ppg_signal[seg[0]:seg[-1]+1]
        is_clean_freq = is_clean_fft(segment, fs)

        is_amp_ok = True
        if use_amp_check:
            seg_amp = np.ptp(segment)
            is_amp_ok = seg_amp <= amp_thresh

        if is_clean_freq and is_amp_ok:
            refined_clean.append(seg)
        else:
            refined_noisy.append(seg)

    return refined_clean, refined_noisy

def adjust_to_local_extrema(ppg_sig, segments, fs):
    maxima = signal.argrelextrema(ppg_sig, np.greater)[0]
    minima = signal.argrelextrema(ppg_sig, np.less)[0]
    extrema = sorted(set(maxima).union(set(minima)))

    adjusted_segments = []
    for seg in segments:
        start, end = seg[0], seg[-1]
        left_extrema = [e for e in extrema if e <= start]
        right_extrema = [e for e in extrema if e >= end]
        new_start = left_extrema[-1] if left_extrema else start
        new_end = right_extrema[0] if right_extrema else end
        adjusted_segments.append(list(range(new_start, new_end + 1)))
    return adjusted_segments
