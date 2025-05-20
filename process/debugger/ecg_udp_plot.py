import socket
import struct
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys
import time
from collections import deque

# === Config ===
UDP_IP = "0.0.0.0"
UDP_PORT = 5000
BUFFER_SIZE = 500  # number of points shown at once

# === Data Buffers ===
ecg_values = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)
timestamps = deque([0]*BUFFER_SIZE, maxlen=BUFFER_SIZE)

# === UDP Setup ===
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# === PyQt App ===
app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(title="Real-Time ECG Monitor")
win.resize(800, 400)
plot = win.addPlot(title="ECG Signal")
plot.setYRange(-2048, 2048)
curve = plot.plot(pen='g')
win.show()

# === Timer Update ===
def update():
    try:
        while True:
            data, _ = sock.recvfrom(1024)
            if len(data) == 8:
                ecg, timestamp = struct.unpack('<iI', data)
                ecg_values.append(ecg)
                timestamps.append(time.time())
    except BlockingIOError:
        pass

    curve.setData(list(ecg_values))

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)  # 50 Hz

# === Run ===
if __name__ == '__main__':
    sys.exit(app.exec_())
