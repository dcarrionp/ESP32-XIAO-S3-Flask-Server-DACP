# Author: Diego Andrés Carrión Portilla (Optimized Version)
# Description: Flask server for ESP32-S3-CAM with motion detection and multiple video streams.

from flask import Flask, render_template, Response
from io import BytesIO
import cv2
import numpy as np
import requests
import time

# ========== CONFIGURACIÓN ==========
app = Flask(__name__)

STREAM_IP = 'http://192.168.18.248'
STREAM_PORT = '81'
STREAM_ROUTE = '/stream'
STREAM_URL = f"{STREAM_IP}:{STREAM_PORT}{STREAM_ROUTE}"

DOWNSCALE_FACTOR = 0.5  # Reducción para mejor rendimiento
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

last_time = time.time()  # Para FPS

# ========== FUNCIONES AUXILIARES ==========

def get_frame():
    """Captura un frame desde el ESP32-S3 y lo reduce."""
    try:
        res = requests.get(STREAM_URL, stream=True, timeout=5)
        for chunk in res.iter_content(chunk_size=100000):
            if len(chunk) > 100:
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                if frame is not None:
                    frame = cv2.resize(frame, (0, 0), fx=DOWNSCALE_FACTOR, fy=DOWNSCALE_FACTOR)
                return frame
    except Exception as e:
        print(f"[ERROR] Error capturando frame: {e}")
        time.sleep(1)
    return None

def encode_frame(frame):
    """Codifica un frame en formato JPEG."""
    success, encoded_image = cv2.imencode(".jpg", frame)
    return bytearray(encoded_image) if success else None

# ========== FUNCIONES DE STREAM ==========

def stream_original():
    """Streaming del video original."""
    while True:
        frame = get_frame()
        if frame is None:
            continue
        encoded = encode_frame(frame)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

def stream_motion():
    """Streaming con detección de movimiento (Adaptive Background Subtraction)."""
    global last_time
    while True:
        frame = get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = fgbg.apply(gray)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        encoded = encode_frame(frame)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

def stream_clahe():
    """Streaming con mejora de iluminación (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    while True:
        frame = get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)

        encoded = encode_frame(enhanced)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

def stream_mask():
    """Streaming solo de la máscara de movimiento."""
    while True:
        frame = get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = fgbg.apply(gray)

        encoded = encode_frame(mask)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

# ========== RUTAS FLASK ==========

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/original_stream")
def original_stream():
    return Response(stream_original(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/motion_stream")
def motion_stream():
    return Response(stream_motion(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/clahe_stream")
def clahe_stream():
    return Response(stream_clahe(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/mask_stream")
def mask_stream():
    return Response(stream_mask(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ========== MAIN ==========

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

