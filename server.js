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

// Initialize files if not exist
if (!fs.existsSync(dataFile)) fs.writeFileSync(dataFile, JSON.stringify([]));
if (!fs.existsSync(userFile)) fs.writeFileSync(userFile, JSON.stringify(null));


// ✅ POST /data — Store new sensor data
app.post('/data', (req, res) => {
  const newData = req.body;
  newData.timestamp = new Date();

  const raw = fs.readFileSync(dataFile);
  const data = JSON.parse(raw);

  data.push(newData);
  fs.writeFileSync(dataFile, JSON.stringify(data, null, 2));

  console.log('📥 Sensor data saved:', newData);

  res.status(200).json({
    success: true,
    message: 'Sensor data saved successfully.'
  });
});


// ✅ GET /data/latest — Get most recent sensor data
app.get('/data/latest', (req, res) => {
  const raw = fs.readFileSync(dataFile);
  const data = JSON.parse(raw);

  if (data.length === 0) {
    return res.status(404).json({
      success: false,
      message: 'No sensor data available.'
    });
  }

  res.json({
    success: true,
    data: data[data.length - 1]
  });
});


// ✅ GET /data/history — Get all sensor data
app.get('/data/history', (req, res) => {
  const raw = fs.readFileSync(dataFile);
  const data = JSON.parse(raw);

  res.json({
    success: true,
    data: data
  });
});


// ✅ POST /user — Save or update user profile
app.post('/user', (req, res) => {
  fs.writeFileSync(userFile, JSON.stringify(req.body, null, 2));
  console.log('👤 User profile updated:', req.body);

  res.status(200).json({
    success: true,
    message: 'User profile saved successfully.'
  });
});


// ✅ GET /user — Fetch user profile
app.get('/user', (req, res) => {
  const raw = fs.readFileSync(userFile);
  const user = JSON.parse(raw);

  if (!user) {
    return res.status(404).json({
      success: false,
      message: 'No user profile found.'
    });
  }

  res.json({
    success: true,
    data: user
  });
});


// ✅ POST /user/reset — Clear user profile
app.post('/user/reset', (req, res) => {
  fs.writeFileSync(userFile, JSON.stringify(null));
  console.log('🔁 User profile reset.');

  res.status(200).json({
    success: true,
    message: 'User profile reset successfully.'
  });
});


// 🚀 Run Server
app.listen(PORT, () => {
  console.log(`✅ PPG Backend running at http://localhost:${PORT}`);
});
