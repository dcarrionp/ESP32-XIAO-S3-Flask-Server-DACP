# Author: Diego Andrés Carrión Portilla (Versión Solo Movimiento)
# Description: Flask + ESP32-S3, solo detección de movimiento adaptativa con FPS

from flask import Flask, render_template, Response
from io import BytesIO

import cv2
import numpy as np
import requests
import time

app = Flask(__name__)

# ========== CONFIGURACIÓN ==========
_URL = 'http://192.168.18.248'  # Cambia por tu IP si es diferente
_PORT = '81'
_ST = '/stream'
SEP = ':'
stream_url = ''.join([_URL, SEP, _PORT, _ST])

# Substracción de fondo adaptativa
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

# Para calcular FPS
last_time = time.time()

# Escala de imagen para mejorar rendimiento
DOWNSCALE_FACTOR = 0.5

# ========== FUNCIONES ==========

def get_frame():
    """Captura un frame de la cámara."""
    try:
        res = requests.get(stream_url, stream=True, timeout=5)
        for chunk in res.iter_content(chunk_size=100000):
            if len(chunk) > 100:
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                frame = cv2.resize(frame, (0, 0), fx=DOWNSCALE_FACTOR, fy=DOWNSCALE_FACTOR)
                return frame
    except Exception as e:
        print(f"Error capturando frame: {e}")
    return None

def encode_frame(frame):
    """Codifica el frame para transmisión."""
    (flag, encodedImage) = cv2.imencode(".jpg", frame)
    if not flag:
        return None
    return bytearray(encodedImage)

def motion_capture():
    """Stream de detección de movimiento con Adaptive Background Subtraction."""
    global last_time
    while True:
        frame = get_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = fgbg.apply(gray)

        # Dibujar rectángulos de movimiento
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Ignorar ruidos pequeños
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Mostrar FPS
        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        encoded = encode_frame(frame)
        if encoded:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

# ========== FLASK ROUTES ==========

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/original_stream")
def original_stream():
    def generate():
        while True:
            frame = get_frame()
            if frame is None:
                continue

            # Aquí no procesamos, solo enviamos el frame original
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag:
                continue

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/clahe_stream")
def clahe_stream():
    def generate():
        # Crear el objeto CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        while True:
            frame = get_frame()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced = clahe.apply(gray)

            (flag, encodedImage) = cv2.imencode(".jpg", enhanced)
            if not flag:
                continue

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/motion_stream")
def motion_stream():
    return Response(motion_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)

