import socket
import struct
import numpy as np
from collections import deque
from PyQt5 import QtWidgets
import pyqtgraph as pg

# === CONFIG ===
UDP_IP = "0.0.0.0"  # Listen on all network interfaces
UDP_PORT = 5000
FS = 50
BUFFER_SIZE = FS * 2  # Buffer for 5 seconds of data

# === Socket Setup ===
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# === Buffers ===
green = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
red = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
ir = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

# === PyQtGraph Setup ===
app = QtWidgets.QApplication([])  # Create PyQt application
win = pg.GraphicsLayoutWidget(title="Live PPG Debug Viewer")  # Create window
win.resize(1000, 800)

# Layout setup with 3 plots for Green, Red, and IR signals
layout = win.ci  # Access the central layout interface

# Plot for Green PPG
p1 = pg.PlotItem(title="Green PPG")
curve1 = p1.plot(pen='g')
layout.addItem(p1, row=0, col=0)

# Plot for Red PPG
p2 = pg.PlotItem(title="Red PPG")
curve2 = p2.plot(pen='r')
layout.addItem(p2, row=1, col=0)

# Plot for IR PPG
p3 = pg.PlotItem(title="IR PPG")
curve3 = p3.plot(pen='c')
layout.addItem(p3, row=2, col=0)

win.show()  # Show window

# === Update Function ===
def update():
    try:
        data, _ = sock.recvfrom(20)  # Receive 20-byte packet from ESP32

        # Slice the first 16 bytes to match the expected 16-byte packet
        data = data[:16]  # Truncate to 16 bytes

        # Unpack the 16-byte packet (Green, Red, IR, Timestamp)
        g, r, i, t = struct.unpack('<iiiI', data)

        # Append the data to respective buffers
        green.append(g)
        red.append(r)
        ir.append(i)

        # Create x-axis (time or sample number)
        x = np.arange(len(green))

        # Update the plots with new data
        curve1.setData(x, green)
        curve2.setData(x, red)
        curve3.setData(x, ir)

        # Set X-axis range for consistent plotting
        if len(green) == BUFFER_SIZE:
            p1.setXRange(x[0], x[-1])
            p2.setXRange(x[0], x[-1])
            p3.setXRange(x[0], x[-1])

    except Exception as e:
        print("Error:", e)

# Set a timer to update the plots every 20 ms (50 Hz)
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)

# === Run App ===
pg.mkQApp().exec_()  # Run the PyQt application
