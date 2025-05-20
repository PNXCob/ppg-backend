import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def plot_abp(abp_signal):
    abp = np.array(abp_signal)
    peaks, _ = find_peaks(abp, distance=50, prominence=10)
    troughs, _ = find_peaks(-abp, distance=50, prominence=10)

    plt.figure(figsize=(10, 4))
    plt.plot(abp, label='ABP waveform', color='red')
    plt.plot(peaks, abp[peaks], 'go', label='SBP Peaks')
    plt.plot(troughs, abp[troughs], 'bo', label='DBP Troughs')
    plt.title('Predicted ABP Waveform')
    plt.xlabel('Sample Index')
    plt.ylabel('Pressure (mmHg)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load from a JSON output file or pasted JSON
    with open("latest_bp.json", "r") as f:
        data = json.load(f)
    
    if "bp_waveform" in data:
        plot_abp(data["bp_waveform"])
    else:
        print("❌ No bp_waveform found in JSON")
