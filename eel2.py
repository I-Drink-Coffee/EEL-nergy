import csv
import logging
import threading
import time
from typing import List
import os
import serial  # For serial communication
import datetime

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import board
import busio
from adafruit_amg88xx import AMG88XX
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression

# Flask app setup
app = Flask(__name__)

# File paths
CSV_FILE = 'temperature_data.csv'
LOG_FILE = 'temperature_log.log'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CSV initialization
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Cycle', 'Average_Temp', 'Predicted_Temp', 'Threshold_Breach'])

# Global variables
TEMP_THRESHOLD = 35.0  # Celsius
sensor_data = []
cycle_data = []
running = True
cycle_count = 0
device_active = False  # Track if the device (e.g., fan) is active

# Serial setup for NodeMCU
try:
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    time.sleep(2)
    logger.info("Serial connection established with NodeMCU")
except Exception as e:
    logger.error(f"Failed to establish serial connection with NodeMCU: {e}")
    exit(1)

try:
    i2c_bus = busio.I2C(board.SCL, board.SDA)
    sensor = AMG88XX(i2c_bus)
    logger.info("AMG88XX sensor initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize AMG88XX sensor: {e}")
    exit(1)

def notify_nodemcu(exceeded: bool):
    """Send serial command to NodeMCU based on threshold status"""
    global device_active
    try:
        if exceeded and not device_active:
            ser.write(b"THRESHOLD_EXCEEDED\n")
            logger.info("Sent THRESHOLD_EXCEEDED command to NodeMCU")
            device_active = True
            response = ser.readline().decode('utf-8').strip()
            if response:
                logger.info(f"NodeMCU response: {response}")
                device_active = False
        elif not exceeded and device_active:
            ser.write(b"RESET_DEVICE\n")
            logger.info("Sent RESET_DEVICE command to NodeMCU")
            device_active = False
            response = ser.readline().decode('utf-8').strip()
            if response:
                logger.info(f"NodeMCU response: {response}")
    except Exception as e:
        logger.error(f"Failed to communicate with NodeMCU: {e}")

def gather_data() -> List[float]:
    """Collect temperature data for 10 minutes"""
    global cycle_count
    temperatures = []
    start_time = time.time()
    logger.info(f"0 - 10 minutes gatherdata (Cycle {cycle_count})")
    try:
        while time.time() - start_time < 600:
            temp_grid = sensor.pixels
            avg_temp = np.mean(temp_grid)
            temperatures.append(avg_temp)
            if avg_temp > TEMP_THRESHOLD:
                logger.warning(f"Temperature exceeded threshold: {avg_temp:.2f}Â°C")
                notify_nodemcu(True)
            else:
                notify_nodemcu(False)
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error during data gathering: {e}")

    avg_temp = np.mean(temperatures) if temperatures else 0
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'),
                        cycle_count,
                        f"{avg_temp:.2f}",
                        "",
                        "Yes" if avg_temp > TEMP_THRESHOLD else "No"])
    return temperatures

def predict_temperature(data: List[float]) -> float:
    """Predict temperature for next 10 minutes using Linear Regression"""
    try:
        X = np.array(range(len(data))).reshape(-1, 1)
        y = np.array(data)
        model = LinearRegression().fit(X, y)
        future_point = len(data) + 600
        prediction = model.predict([[future_point]])[0]
        return prediction
    except Exception as e:
        logger.error(f"Error in temperature prediction: {e}")
        return 0.0

def data_collection_cycle():
    global sensor_data, cycle_data, cycle_count
    while running:
        hour_start = time.time()
        for _ in range(6):
            cycle_count += 1
            cycle_start = time.time()
            temp_data = gather_data()
            sensor_data = temp_data
            predicted_temp = predict_temperature(temp_data)
            avg_temp = np.mean(temp_data) if temp_data else 0
            cycle_data.append({
                'cycle': cycle_count,
                'avg_temp': avg_temp,
                'predicted_temp': predicted_temp
            })
            logger.info(f">>>> predicted temp (10 minutes): {predicted_temp:.2f}Â°C (Cycle {cycle_count})")
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'),
                                cycle_count,
                                f"{avg_temp:.2f}",
                                f"{predicted_temp:.2f}",
                                ""])
            if predicted_temp > TEMP_THRESHOLD:
                logger.warning(f"Predicted temperature exceeds threshold: {predicted_temp:.2f}Â°C")
                notify_nodemcu(True)
            else:
                notify_nodemcu(False)
            cycle_duration = time.time() - cycle_start
            if cycle_duration < 600:
                time.sleep(600 - cycle_duration)
        elapsed = time.time() - hour_start
        if elapsed < 3600:
            time.sleep(3600 - elapsed)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    global sensor_data, cycle_data, cycle_count
    return jsonify({
        'cycle': cycle_count,
        'current_avg': np.mean(sensor_data) if sensor_data else 0,
        'latest_predicted': cycle_data[-1]['predicted_temp'] if cycle_data else 0,
        'history': cycle_data
    })

def generate_temperature_feed():
    while True:
        try:
            frame = np.array(sensor.pixels)
            frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
            frame = frame.astype(np.uint8)
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_CUBIC)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Error generating video feed: {e}")
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_temperature_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def cleanup_files():
    """Delete log and CSV if older than 30 days"""
    while True:
        now = time.time()
        for file in [CSV_FILE, LOG_FILE]:
            if os.path.exists(file):
                file_age_days = (now - os.path.getmtime(file)) / 86400
                if file_age_days > 30:
                    try:
                        os.remove(file)
                        logger.info(f"Deleted old file: {file}")
                    except Exception as e:
                        logger.error(f"Error deleting file {file}: {e}")
        time.sleep(86400)  # Run daily

def start_background_thread():
    thread = threading.Thread(target=data_collection_cycle)
    thread.daemon = True
    thread.start()
    cleanup_thread = threading.Thread(target=cleanup_files)
    cleanup_thread.daemon = True
    cleanup_thread.start()

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    with open('templates/index.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Temperature Monitoring Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; text-align: center; padding: 20px; }
        h1 { color: #333; }
        #video { border: 4px solid #333; border-radius: 8px; margin: 20px auto; display: block; }
        #data { margin-top: 30px; font-size: 1.2em; color: #444; }
        table { margin: 20px auto; border-collapse: collapse; }
        th, td { padding: 8px 12px; border: 1px solid #aaa; }
        th { background-color: #eee; }
    </style>
</head>
<body>
    <h1>Real-Time Temperature Monitoring</h1>
    <img id="video" src="/video_feed" alt="Live Heatmap" width="320" height="320">
    <div id="data">
        <p><strong>Cycle:</strong> <span id="cycle">Loading...</span></p>
        <p><strong>Current Average Temperature:</strong> <span id="current_avg">Loading...</span> Â°C</p>
        <p><strong>Predicted Temperature (10 mins):</strong> <span id="predicted">Loading...</span> Â°C</p>
        <h2>Cycle History</h2>
        <table id="history-table">
            <thead>
                <tr>
                    <th>Cycle</th>
                    <th>Avg Temp (Â°C)</th>
                    <th>Predicted Temp (Â°C)</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>
    <script>
        function fetchData() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('cycle').textContent = data.cycle;
                    document.getElementById('current_avg').textContent = data.current_avg.toFixed(2);
                    document.getElementById('predicted').textContent = data.latest_predicted.toFixed(2);
                    const historyTable = document.querySelector('#history-table tbody');
                    historyTable.innerHTML = '';
                    data.history.forEach(entry => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${entry.cycle}</td>
                            <td>${entry.avg_temp.toFixed(2)}</td>
                            <td>${entry.predicted_temp.toFixed(2)}</td>
                        `;
                        historyTable.appendChild(row);
                    });
                })
                .catch(error => console.error('Error fetching data:', error));
        }
        setInterval(fetchData, 5000);
        fetchData();
    </script>
</body>
</html>''')
    logger.info("Starting temperature monitoring application...")
    start_background_thread()
    app.run(host='0.0.0.0', port=5000, debug=False)