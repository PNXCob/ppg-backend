const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(bodyParser.json());

// File paths
const dataFile = path.join(__dirname, 'data.json');
const userFile = path.join(__dirname, 'user.json');

// Initialize files if not present
if (!fs.existsSync(dataFile)) fs.writeFileSync(dataFile, JSON.stringify([]));
if (!fs.existsSync(userFile)) fs.writeFileSync(userFile, JSON.stringify(null));

// ✅ POST /data — Save sensor data
app.post('/data', (req, res) => {
  const newData = req.body;
  newData.timestamp = new Date();

  const raw = fs.readFileSync(dataFile);
  const data = JSON.parse(raw);
  data.push(newData);

  fs.writeFileSync(dataFile, JSON.stringify(data, null, 2));
  console.log('📥 Sensor data saved:', newData);

  res.status(200).json({ message: 'Data stored' });
});

// ✅ GET /data/latest — Get the most recent entry
app.get('/data/latest', (req, res) => {
  const raw = fs.readFileSync(dataFile);
  const data = JSON.parse(raw);

  if (data.length === 0) {
    return res.status(404).json({ message: 'No data available' });
  }

  res.json(data[data.length - 1]);
});

// ✅ GET /data/history — Get all saved sensor data
app.get('/data/history', (req, res) => {
  const raw = fs.readFileSync(dataFile);
  const data = JSON.parse(raw);
  res.json(data);
});

// ✅ POST /user — Save or update user profile
app.post('/user', (req, res) => {
  fs.writeFileSync(userFile, JSON.stringify(req.body, null, 2));
  console.log('👤 User profile updated:', req.body);
  res.status(200).json({ message: 'User profile saved' });
});

// ✅ GET /user — Retrieve user profile
app.get('/user', (req, res) => {
  const raw = fs.readFileSync(userFile);
  const user = JSON.parse(raw);

  if (!user) {
    return res.status(404).json({ message: 'No user profile' });
  }

  res.json(user);
});

// ✅ POST /user/reset — Clear user profile
app.post('/user/reset', (req, res) => {
  fs.writeFileSync(userFile, JSON.stringify(null));
  console.log('🔁 User profile has been reset.');
  res.status(200).json({ message: 'User profile reset' });
});

// 🚀 Start server
app.listen(PORT, () => {
  console.log(`✅ PPG Backend is live at http://localhost:${PORT}`);
});
