import socket
import struct
import sys
import numpy as np
from collections import deque
from PyQt5 import QtWidgets
import pyqtgraph as pg
import pywt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

# --- UDP Config ---
UDP_IP = "0.0.0.0"
UDP_PORT = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# --- Settings ---
FS = 50
BUFFER_SECONDS = 30
BUFFER_SIZE = FS * BUFFER_SECONDS

# --- Buffers ---
green_values = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
timestamps = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

# --- PyQtGraph Setup ---
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Real-Time Respiratory Rate Debugger (BW Method)")
win.resize(1000, 500)

# Plot 1: Raw Green PPG
p1 = win.addPlot(title="Raw Green PPG")
curve_raw = p1.plot(pen='g')

# Plot 2: Baseline Wander + Peaks
win.nextRow()
p2 = win.addPlot(title="Baseline Wander (Wavelet)")
curve_bw = p2.plot(pen='b')
curve_peaks = p2.plot(pen=None, symbol='o', symbolSize=6, symbolBrush='r')

win.show()

# --- BW RR Function ---
def compute_bw(signal):
    coeffs = pywt.wavedec(signal, 'sym4', level=7)
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])
    baseline = pywt.waverec(coeffs, 'sym4')[:len(signal)]
    smoothed = savgol_filter(baseline, 51, 3)
    peaks, _ = find_peaks(smoothed, distance=FS*2, prominence=25)
    rr_bpm = len(peaks) * (60 / BUFFER_SECONDS)
    return smoothed, peaks, rr_bpm

# --- Data Update ---
def update_stream():
    try:
        data, _ = sock.recvfrom(16)  # match ESP32 packet size
        if len(data) == 16:
            green = struct.unpack("i", data[0:4])[0]
            ts = struct.unpack("I", data[12:16])[0]  # timestamp in ms
            green_values.append(green)
            timestamps.append(ts / 1000.0)
            t = np.array(timestamps) - timestamps[0]
            curve_raw.setData(t, green_values)
    except Exception as e:
        print("⚠️ UDP Read Error:", e)

# --- RR Estimation + Display ---
def update_rr():
    if len(green_values) < BUFFER_SIZE:
        print("⏳ Waiting for 1500 samples...")
        return

    raw = np.array(green_values)
    raw_time = np.array(timestamps)
    t_aligned = raw_time - raw_time[0]
    t_uniform = np.linspace(t_aligned[0], t_aligned[-1], BUFFER_SIZE)
    uniform_signal = interp1d(t_aligned, raw, fill_value="extrapolate")(t_uniform)

    bw, peaks, rr = compute_bw(uniform_signal)

    curve_bw.setData(t_uniform, bw)
    curve_peaks.setData(t_uniform[peaks], bw[peaks])

    print(f"🫁 Real-Time RR = {rr:.2f} bpm")
    win.setWindowTitle(f"RR Debugger | RR = {rr:.2f} bpm")

# --- Timers ---
timer_stream = pg.QtCore.QTimer()
timer_stream.timeout.connect(update_stream)
timer_stream.start(1)

timer_rr = pg.QtCore.QTimer()
timer_rr.timeout.connect(update_rr)
timer_rr.start(BUFFER_SECONDS * 1000)

# --- Launch ---
import sys
sys.exit(app.exec_())
