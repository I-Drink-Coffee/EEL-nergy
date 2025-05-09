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
from flask import Flask, render_template, Response, jsonify,  send_file, redirect, make_response
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
@app.route('/generate_204')
def redirect_android():
    resp = make_response('', 302)
    resp.headers['Location'] = 'http://192.168.4.1:5000'
    return resp
@app.route('/plot.png')
def plot_png():
    fig, ax = plt.subplots()
    x = [entry['cycle'] for entry in cycle_data]
    y = [entry['avg_temp'] for entry in cycle_data]
    ax.plot(x, y, marker='o')
    ax.set_title("Avg Temperature Over Cycles")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Temp (Â°C)")
    ax.grid(True)
    output = io.BytesIO()
    fig.savefig(output, format='png')
    plt.close(fig)
    output.seek(0)
    return send_file(output, mimetype='image/png')


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
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />

    <title>Temperature Monitoring Dashboard</title>

    <style>
        :root {
            --primary-color: #1a4971;   /* Deep ocean blue  */
            --secondary-color: #2a9d8f; /* Vibrant teal      */
            --background-color: #e9f6fb;/* Soft sea‑foam     */
            --border-color: #b3d9e6;    /* Light blue        */
            --table-header-bg: #c4e4f0; /* Pale aqua         */
            --accent-color: #e76f51;    /* Coral reef        */
            --shadow-color: rgba(0, 0, 0, 0.15);
            --card-bg: rgba(255, 255, 255, 0.95);
        }

        /* ---------- Base resets ---------- */
        *, *::before, *::after { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(145deg, var(--background-color), #d4e9f2);
            color: var(--secondary-color);
            line-height: 1.6;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding: 1rem;
            overflow-x: hidden;
            scroll-behavior: smooth;
        }
        img { max-width: 100%; height: auto; }

        /* ---------- Header ---------- */
        header {
            position: sticky; top: 0; z-index: 100;
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            border-radius: 0 0 16px 16px;
            box-shadow: 0 4px 12px var(--shadow-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 {
            margin: 0;
            color: var(--primary-color);
            font-size: clamp(1.8rem, 5vw, 2.2rem);
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .refresh-btn {
            background: var(--secondary-color);
            color: #fff;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: background 0.3s, transform 0.2s;
        }
        .refresh-btn:hover   { background: #21867a; transform: translateY(-2px); }
        .refresh-btn::before { content: '↻'; }

        /* ---------- Main grid ---------- */
        main {
            max-width: 1600px;
            margin: 2rem auto;
            display: grid;
            gap: 2rem;
            grid-template-columns: 1fr minmax(320px, 400px) 1fr;
            padding: 0 1rem;
        }

        /* ---------- Heatmap ---------- */
        .heatmap {
            grid-column: 2 / 3;
            background: var(--card-bg);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 6px 16px var(--shadow-color);
            position: relative;
            overflow: hidden;
            animation: fadeIn 0.6s ease-out forwards;
        }
        .heatmap::before {
            content: '';
            position: absolute; inset: 0 0 auto;
            height: 4px;
            background: linear-gradient(90deg, var(--secondary-color), var(--accent-color));
        }
        .heatmap img {
            aspect-ratio: 1/1;
            object-fit: cover;
            border-radius: 12px;
            border: 3px solid var(--primary-color);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .heatmap img:hover { transform: scale(1.03); box-shadow: 0 8px 20px var(--shadow-color); }

        /* ---------- Data section ---------- */
        .data-section {
            grid-column: 1 / 4;
            background: var(--card-bg);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 6px 16px var(--shadow-color);
            display: flex;
            flex-direction: column;
            align-items: center;
            animation: fadeIn 0.6s ease-out forwards;
        }
        .metrics {
            width: 100%;
            max-width: 1000px;
            margin-bottom: 2rem;
            display: grid;
            gap: 1rem;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }
        .metric-card {
            background: #f0f8fc;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .metric-card:hover   { transform: translateY(-4px); box-shadow: 0 4px 12px var(--shadow-color); }
        .metric-card strong  { color: var(--primary-color); font-weight: 600; margin-bottom: .5rem; display: block; }
        .metric-card span    { font-size: 1.2rem; color: var(--secondary-color); }

        /* ---------- Headings ---------- */
        h2 {
            color: var(--primary-color);
            font-size: clamp(1.4rem, 3.5vw, 1.6rem);
            font-weight: 600;
            margin: 1.5rem 0;
            position: relative;
            display: inline-block;
        }
        h2::after {
            content: '';
            position: absolute; bottom: -4px; left: 0;
            width: 100%; height: 2px;
            background: var(--accent-color);
        }

        /* ---------- Table ---------- */
        table {
            width: min(100%, 1000px);
            border-collapse: separate;
            border-spacing: 0;
            background: #fff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px var(--shadow-color);
            margin: 1rem 0;
            display: block;         /* mobile horizontal scroll */
            overflow-x: auto;
        }
        thead { width: 100%; }
        th, td {
            padding: 1rem;
            border: 1px solid var(--border-color);
            text-align: center;
            font-size: clamp(.9rem, 2vw, 1rem);
        }
        th {
            background: var(--table-header-bg);
            color: var(--primary-color);
            font-weight: 600;
        }
        tr:nth-child(even) { background: #f0f8fc; }
        tr:hover           { background: #e0f0fa; }

        /* ---------- Plot section ---------- */
        .plot-section {
            grid-column: 1 / 4;
            background: var(--card-bg);
            margin-bottom: 2rem;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 6px 16px var(--shadow-color);
            display: flex;
            justify-content: center;
            animation: fadeIn 0.6s ease-out forwards;
        }
        .plot-section img {
            max-width: 800px;
            border-radius: 12px;
            border: 2px solid var(--border-color);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .plot-section img:hover { transform: scale(1.02); box-shadow: 0 8px 20px var(--shadow-color); }

        /* ---------- Footer ---------- */
        footer {
            text-align: center;
            padding: 1rem;
            color: var(--primary-color);
            font-size: .9rem;
            margin-top: auto;
        }

        /* ---------- Media queries ---------- */
        @media (max-width: 1200px) {
            main              { grid-template-columns: 1fr; }
            .heatmap,
            .data-section,
            .plot-section     { grid-column: 1 / 2; }
            .metrics          { grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); }
        }
        @media (max-width: 768px) {
            body              { padding: .5rem; }
            header            { padding: .75rem 1rem; flex-direction: column; gap: .5rem; }
            main              { margin: 1rem auto; gap: 1rem; }
            .data-section     { padding: 1rem; }
            .metric-card span { font-size: 1.1rem; }
        }
        @media (max-width: 480px) {
            h1                { font-size: 1.6rem; }
            h2                { font-size: 1.2rem; }
            .metric-card      { padding: .75rem; }
            .metric-card span { font-size: 1rem; }
            .refresh-btn      { width: 100%; justify-content: center; }
            th, td            { padding: .5rem; }
        }

        /* ---------- Animations ---------- */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0);   }
        }
    </style>
</head>

<body>
    <!-- ---------- Header ---------- -->
    <header>
        <h1>Real‑Time Temperature Monitoring</h1>
        <button class="refresh-btn" onclick="fetchData(); updatePlot();">Refresh</button>
    </header>

    <!-- ---------- Main ---------- -->
    <main>
        <!-- Heatmap -->
        <section class="heatmap" aria-label="Live heatmap display">
            <img id="video" src="/video_feed" alt="Live Heatmap" />
        </section>

        <!-- Data & Plot -->
        <section class="data-section" aria-label="Temperature data">
            <!-- KPI cards -->
            <div class="metrics">
                <div class="metric-card">
                    <strong>Cycle</strong>
                    <span id="cycle">Loading…</span>
                </div>
                <div class="metric-card">
                    <strong>Current Avg Temp</strong>
                    <span id="current_avg">Loading…</span> °C
                </div>
                <div class="metric-card">
                    <strong>Predicted Temp (10 mins)</strong>
                    <span id="predicted">Loading…</span> °C
                </div>
            </div>

            <!-- History table -->
            <h2>Cycle History</h2>
            <table id="history-table" aria-label="Temperature history table">
                <thead>
                    <tr>
                        <th scope="col">Cycle</th>
                        <th scope="col">Avg Temp (°C)</th>
                        <th scope="col">Predicted Temp (°C)</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>

            <!-- Plot -->
            <h2>Average Temperature Plot</h2>
            <div class="plot-section">
                <img id="temp-plot" src="/plot.png" alt="Temperature Plot" />
            </div>
        </section>
    </main>

    <!-- ---------- Footer ---------- -->
    <footer>© 2025 Temperature Monitoring System.</footer>

    <!-- ---------- JS ---------- -->
    <script>
        function fetchData() {
            fetch('/data')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('cycle').textContent        = data.cycle;
                    document.getElementById('current_avg').textContent  = data.current_avg.toFixed(2);
                    document.getElementById('predicted').textContent    = data.latest_predicted.toFixed(2);

                    const tbody = document.querySelector('#history-table tbody');
                    tbody.innerHTML = '';
                    data.history.forEach(entry => {
                        tbody.insertAdjacentHTML('beforeend', `
                            <tr>
                                <td>${entry.cycle}</td>
                                <td>${entry.avg_temp.toFixed(2)}</td>
                                <td>${entry.predicted_temp.toFixed(2)}</td>
                            </tr>`);
                    });
                })
                .catch(console.error);
        }

        function updatePlot() {
            document.getElementById('temp-plot').src = '/plot.png?rand=' + Math.random();
        }

        /* Auto‑refresh every 5 s */
        setInterval(() => { fetchData(); updatePlot(); }, 5000);
        /* Initial load */
        fetchData(); updatePlot();
    </script>
</body>
</html>
''')
    logger.info("Starting temperature monitoring application...")
    start_background_thread()
    app.run(host='0.0.0.0', port=5000, debug=False)