const dgram = require('dgram');
const express = require('express');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const http = require('http');

// === CONFIG ===
const UDP_PORT = 5000;
const HTTP_PORT = 3000;
const FS = 50;
const HR_WINDOW_SIZE = 750;
const SPO2_WINDOW_SIZE = 500;
const BP_WINDOW_SIZE = 1024;
const RR_WINDOW_SIZE = 750;

const SIGNAL_PTP_THRESHOLD = 300;
const SIGNAL_MAX_THRESHOLD = 1000;

// === Setup ===
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
const udpServer = dgram.createSocket('udp4');

// === Buffers ===
let greenBuffer = [], redBuffer = [], irBuffer = [], timestampBuffer = [];
let lastActiveTime = Date.now();

// === Temp Buffering ===
let lastTemp = null;
let lastTempLogTime = 0;

// === Guards (1 per vital) ===
let isProcessingHR = false;
let isProcessingSPO2 = false;
let isProcessingBP = false;
let isProcessingRR = false;

// === Broadcast Helper ===
function broadcast(data) {
  const json = JSON.stringify(data);
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(json);
    }
  });
}

// === Signal Validation ===
function isInvalidSignal(arr, ptpThreshold = SIGNAL_PTP_THRESHOLD, maxThreshold = SIGNAL_MAX_THRESHOLD) {
  const max = Math.max(...arr);
  const min = Math.min(...arr);
  const ptp = max - min;
  return ptp < ptpThreshold || max < maxThreshold;
}

// === UDP Input ===
udpServer.on('message', (msg) => {
  if (msg.length !== 20) return;
  const green = msg.readInt32LE(0);
  const red = msg.readInt32LE(4);
  const ir = msg.readInt32LE(8);
  const temp = msg.readInt32LE(12) / 100;
  const timestamp = msg.readUInt32LE(16) / 1000.0;

  if (Math.abs(green) > 30 || Math.abs(red) > 30 || Math.abs(ir) > 30) {
    lastActiveTime = Date.now();
  }

  greenBuffer.push(green);
  redBuffer.push(red);
  irBuffer.push(ir);
  timestampBuffer.push(timestamp);
  lastTemp = temp;

  const now = Date.now();
  if (now - lastTempLogTime > 2000) {
    console.log(`🌡️ Temp: ${temp.toFixed(2)}°C`);
    lastTempLogTime = now;
  }

  broadcast({ green, red, ir, temp, timestamp });

  tryHR();
  trySPO2();
  tryBP();
  tryRR();
});

// === No-hand Auto-Clear ===
setInterval(() => {
  if (Date.now() - lastActiveTime > 10000) {
    greenBuffer = [];
    redBuffer = [];
    irBuffer = [];
    timestampBuffer = [];
    console.warn("🚨 No hand detected — buffers cleared.");
    broadcast({ fallback: true, message: "No hand detected", timestamp: new Date().toISOString() });
  }
}, 2000);

// === Real-time Triggered Vitals ===

function tryHR() {
  if (isProcessingHR || greenBuffer.length < HR_WINDOW_SIZE) return;
  const greens = greenBuffer.slice(-HR_WINDOW_SIZE);
  const times = timestampBuffer.slice(-HR_WINDOW_SIZE);
  if (isInvalidSignal(greens)) return;

  isProcessingHR = true;
  const py = spawn('python', ['process/heartrate.py']);
  py.stdin.write(JSON.stringify({ ppg: greens, timestamps: times }));
  py.stdin.end();

  let output = '';
  py.stdout.on('data', (data) => output += data.toString());
  py.stderr.on('data', (err) => console.error('❌ HR error:', err.toString()));
  py.on('close', () => {
    try {
      const result = JSON.parse(output);
      if (result?.bpm !== undefined) {
        console.log(`✅ BPM: ${result.bpm}`);
        broadcast({ bpm: result.bpm, timestamp: new Date().toISOString() });
      } else {
        console.warn('⚠️ Invalid HR result:', output);
      }
    } catch (e) {
      console.error('❌ Failed to parse HR output:', e.message);
    }
    isProcessingHR = false;
  });
}

function trySPO2() {
  if (isProcessingSPO2 || redBuffer.length < SPO2_WINDOW_SIZE || irBuffer.length < SPO2_WINDOW_SIZE) return;
  const reds = redBuffer.slice(-SPO2_WINDOW_SIZE);
  const irs = irBuffer.slice(-SPO2_WINDOW_SIZE);
  const times = timestampBuffer.slice(-SPO2_WINDOW_SIZE);
  if (isInvalidSignal(reds) || isInvalidSignal(irs)) return;

  isProcessingSPO2 = true;
  const py = spawn('python', ['process/spo2.py']);
  py.stdin.write(JSON.stringify({ red: reds, ir: irs, timestamps: times }));
  py.stdin.end();

  let output = '';
  py.stdout.on('data', (data) => output += data.toString());
  py.stderr.on('data', (err) => console.error('❌ SpO₂ error:', err.toString()));
  py.on('close', () => {
    try {
      const result = JSON.parse(output);
      if (result?.spo2 !== undefined) {
        console.log(`✅ SpO₂: ${result.spo2}%`);
        broadcast({ spo2: result.spo2, timestamp: new Date().toISOString() });
      } else {
        console.warn('⚠️ Invalid SpO₂ result:', output);
      }
    } catch (e) {
      console.error('❌ Failed to parse SpO₂ output:', e.message);
    }
    isProcessingSPO2 = false;
  });
}

function tryBP() {
  if (isProcessingBP || greenBuffer.length < BP_WINDOW_SIZE) return;
  const greens = greenBuffer.slice(-BP_WINDOW_SIZE);
  const times = timestampBuffer.slice(-BP_WINDOW_SIZE);
  if (isInvalidSignal(greens)) return;

  isProcessingBP = true;
  const py = spawn('python', ['process/blood_pressure.py']);
  py.stdin.write(JSON.stringify({ green: greens, timestamps: times }));
  py.stdin.end();

  let output = '';
  py.stdout.on('data', (data) => output += data.toString());
  py.stderr.on('data', (err) => console.error('❌ BP error:', err.toString()));
  py.on('close', () => {
    try {
      const result = JSON.parse(output);
      if (result?.sbp !== undefined && result?.dbp !== undefined) {
        console.log(`✅ BP: ${result.sbp}/${result.dbp} mmHg`);
        broadcast({ sbp: result.sbp, dbp: result.dbp, timestamp: new Date().toISOString() });
      } else {
        console.warn('⚠️ Invalid BP result:', output);
      }
    } catch (e) {
      console.error('❌ Failed to parse BP output:', e.message);
    }
    isProcessingBP = false;
  });
}

function tryRR() {
  if (isProcessingRR || greenBuffer.length < RR_WINDOW_SIZE) return;
  const greens = greenBuffer.slice(-RR_WINDOW_SIZE);
  const times = timestampBuffer.slice(-RR_WINDOW_SIZE);
  if (isInvalidSignal(greens)) return;

  isProcessingRR = true;
  const py = spawn('python', ['process/respiratory_rate.py']);
  py.stdin.write(JSON.stringify({ green: greens, timestamps: times }));
  py.stdin.end();

  let output = '';
  py.stdout.on('data', (data) => output += data.toString());
  py.stderr.on('data', (err) => console.error('❌ RR error:', err.toString()));
  py.on('close', () => {
    try {
      const result = JSON.parse(output);
      if (result?.rr !== undefined) {
        console.log(`✅ RR: ${result.rr} bpm`);
        broadcast({ rr: result.rr, timestamp: new Date().toISOString() });
      } else {
        console.warn('⚠️ Invalid RR result:', output);
      }
    } catch (e) {
      console.error('❌ Failed to parse RR output:', e.message);
    }
    isProcessingRR = false;
  });
}

// === Start Servers ===
udpServer.bind(UDP_PORT, () => {
  console.log(`📡 UDP Server listening on port ${UDP_PORT}`);
});
server.listen(HTTP_PORT, () => {
  console.log(`🌐 HTTP + WebSocket running at http://localhost:${HTTP_PORT}`);
});
app.get('/', (req, res) => {
  res.send('👋 Biomnitrix Vital Monitor Active');
});
