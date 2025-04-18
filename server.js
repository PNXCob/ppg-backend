// ... keep all previous imports
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const dgram = require('dgram');
const { execFile } = require('child_process');
const WebSocket = require('ws');

const app = express();
const PORT = process.env.PORT || 3000;
const UDP_PORT = 5000;
const WS_PORT = 8080;

app.use(cors());
app.use(bodyParser.json());

const dataFile = path.join(__dirname, 'data.json');
const userFile = path.join(__dirname, 'user.json');
if (!fs.existsSync(dataFile)) fs.writeFileSync(dataFile, JSON.stringify([]));
if (!fs.existsSync(userFile)) fs.writeFileSync(userFile, JSON.stringify(null));

// === WebSocket Server ===
const wss = new WebSocket.Server({ port: WS_PORT });
console.log(`WebSocket ready on ws://localhost:${WS_PORT}`);

let lastUDPTime = Date.now();
let fallbackSent = false;

wss.on('connection', (ws) => {
  console.log("📡 Client connected");

  ws.on('message', (msg) => {
    try {
      const parsed = JSON.parse(msg);
      if (parsed.type === 'ping') {
        console.log("📶 Ping received from client at", new Date().toISOString());
      }
    } catch (e) {
      console.log("📥 Raw message received:", msg);
    }
  });

  ws.on('close', () => {
    console.log("⚠️ Client disconnected");
  });

  ws.on('error', (err) => {
    console.error("❌ WebSocket server error:", err);
  });
});

function broadcast(data) {
  const json = JSON.stringify(data);
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) client.send(json);
  });
}

function broadcastFallback() {
  if (!fallbackSent && Date.now() - lastUDPTime > 15000) {
    const raw = fs.readFileSync(dataFile);
    const data = JSON.parse(raw);
    if (data.length > 0) {
      console.log("📭 No incoming UDP. Broadcasting last known data.");
      broadcast(data[data.length - 1]);
      fallbackSent = true;
    }
  }
}

setInterval(broadcastFallback, 5000);

// === UDP Setup ===
const udpServer = dgram.createSocket('udp4');
const buffer = [];
const SAMPLE_RATE = 50;
const WINDOW_SECONDS = 10;
const MAX_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS;

udpServer.on('message', (msg) => {
  if (msg.length !== 20) return;

  const green = msg.readInt32LE(0);
  const red = msg.readInt32LE(4);
  const ir = msg.readInt32LE(8);
  const timestamp = msg.readUInt32LE(12);
  const temp = msg.readFloatLE(16);

  buffer.push({ green, red, ir, timestamp, temp });
  lastUDPTime = Date.now();
  fallbackSent = false;

  if (buffer.length >= MAX_SAMPLES) {
    const greenArray = buffer.map(e => e.green);
    const redArray = buffer.map(e => e.red);
    const irArray = buffer.map(e => e.ir);
    const tsArray = buffer.map(e => e.timestamp / 1000);
    const latestTemp = buffer[buffer.length - 1].temp.toFixed(1);

    buffer.length = 0;
    const results = {};

    execFile('python3', ['process/blood_pressure.py', JSON.stringify(greenArray), JSON.stringify(tsArray)], (err, stdout) => {
      if (!err) results.bp = JSON.parse(stdout);

      execFile('python3', ['process/heartrate.py', JSON.stringify(greenArray), JSON.stringify(tsArray)], (err, stdout) => {
        if (!err) results.hr = JSON.parse(stdout);

        execFile('python3', ['process/respiratory_rate.py', JSON.stringify(greenArray), JSON.stringify(tsArray)], (err, stdout) => {
          if (!err) results.rr = JSON.parse(stdout);

          execFile('python3', ['process/spo2.py', JSON.stringify(redArray), JSON.stringify(irArray), JSON.stringify(tsArray)], (err, stdout) => {
            if (!err) results.spo2 = JSON.parse(stdout);

            results.timestamp = new Date().toISOString();
            results.temp = parseFloat(latestTemp);

            const raw = fs.readFileSync(dataFile);
            const data = JSON.parse(raw);
            data.push(results);
            fs.writeFileSync(dataFile, JSON.stringify(data, null, 2));

            broadcast(results);
            console.log(`[WS] Sent vitals @ ${results.timestamp}`);
          });
        });
      });
    });
  }
});

udpServer.bind(UDP_PORT, () => {
  console.log(`UDP listening on port ${UDP_PORT}`);
});

// === API Routes ===
app.get('/data/latest', (req, res) => {
  const raw = fs.readFileSync(dataFile);
  const data = JSON.parse(raw);
  if (data.length === 0) return res.status(404).json({ success: false, message: 'No data.' });
  res.json({ success: true, data: data[data.length - 1] });
});

app.get('/data/history', (req, res) => {
  const raw = fs.readFileSync(dataFile);
  res.json(JSON.parse(raw));
});

app.post('/user', (req, res) => {
  fs.writeFileSync(userFile, JSON.stringify(req.body, null, 2));
  res.json({ success: true, message: 'User saved.' });
});

app.get('/user', (req, res) => {
  const raw = fs.readFileSync(userFile);
  const user = JSON.parse(raw);
  if (!user) return res.status(404).json({ success: false, message: 'User not found' });
  res.json({ success: true, data: user });
});

app.post('/user/reset', (req, res) => {
  fs.writeFileSync(userFile, JSON.stringify(null));
  res.json({ success: true, message: 'User reset.' });
});

app.listen(PORT, () => {
  console.log(`HTTP API running on http://localhost:${PORT}`);
});
