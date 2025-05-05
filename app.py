# Author: Diego Andrés Carrión Portilla (Optimized Version)
# Description: Flask server for ESP32-S3-CAM with motion detection and multiple video streams.

from flask import Flask, render_template, Response, request
from io import BytesIO
import cv2
import numpy as np
import requests
import time

app = Flask(__name__)

# ========== CONFIGURACIÓN ==========
STREAM_IP = 'http://192.168.18.248'
STREAM_PORT = '81'
STREAM_ROUTE = '/stream'
STREAM_URL = f"{STREAM_IP}:{STREAM_PORT}{STREAM_ROUTE}"

DOWNSCALE_FACTOR = 0.5
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)
last_time = time.time()

# ========== FUNCIONES AUXILIARES ==========
def get_frame():
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
        print(f"[ERROR] Capturando frame: {e}")
        time.sleep(1)
    return None

def encode_frame(frame):
    success, encoded_image = cv2.imencode(".jpg", frame)
    return bytearray(encoded_image) if success else None

# ========== FUNCIONES DE STREAM ==========
def stream_original():
    while True:
        frame = get_frame()
        if frame is None:
            continue
        encoded = encode_frame(frame)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

def stream_motion():
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

def stream_equalized():
    while True:
        frame = get_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        encoded = encode_frame(equalized)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

def stream_gamma():
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")

    while True:
        frame = get_frame()
        if frame is None:
            continue
        adjusted = cv2.LUT(frame, table)

        encoded = encode_frame(adjusted)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

def stream_mask():
    while True:
        frame = get_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = fgbg.apply(gray)

        encoded = encode_frame(mask)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

def stream_and():
    while True:
        frame = get_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = fgbg.apply(gray)
        frame_and = cv2.bitwise_and(frame, frame, mask=mask)

        encoded = encode_frame(frame_and)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

def stream_or():
    while True:
        frame = get_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = fgbg.apply(gray)
        frame_or = cv2.bitwise_or(frame, frame, mask=mask)

        encoded = encode_frame(frame_or)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

def stream_xor():
    while True:
        frame = get_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = fgbg.apply(gray)
        frame_xor = cv2.bitwise_xor(frame, frame, mask=mask)

        encoded = encode_frame(frame_xor)
        if encoded:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

# ========== FUNCIONES DE RUIDO ==========

def add_gaussian_noise(image, mean=0, std=20):
    """Agrega ruido gaussiano a una imagen a color."""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std, (row, col, ch)).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy

def add_speckle_noise(image, var=0.04):
    """Agrega ruido speckle a una imagen a color."""
    row, col, ch = image.shape
    noise = np.random.randn(row, col, ch) * var
    noisy = image + image * noise
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy


# ========== RUTAS FLASK ==========
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/original_stream")
def original_stream():
    return Response(stream_original(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/motion_stream")
def motion_stream():
    return Response(stream_motion(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/clahe_stream")
def clahe_stream():
    return Response(stream_clahe(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/equalized_stream")
def equalized_stream():
    return Response(stream_equalized(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/gamma_stream")
def gamma_stream():
    return Response(stream_gamma(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/mask_stream")
def mask_stream():
    return Response(stream_mask(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/and_stream")
def and_stream():
    return Response(stream_and(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/or_stream")
def or_stream():
    return Response(stream_or(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/xor_stream")
def xor_stream():
    return Response(stream_xor(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/gaussian_noise_stream")
def gaussian_noise_stream():
    """Streaming con ruido gaussiano configurable"""
    mean = int(request.args.get('mean', 0))   # Default 0 si no viene nada
    std = int(request.args.get('std', 20))     # Default 20 si no viene nada

    def generate():
        while True:
            frame = get_frame()
            if frame is None:
                continue

            noisy_frame = add_gaussian_noise(frame, mean, std)
            encoded = encode_frame(noisy_frame)
            if encoded:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/speckle_noise_stream")
def speckle_noise_stream():
    """Streaming con ruido speckle configurable"""
    var = float(request.args.get('var', 0.04))  # Default 0.04 si no viene nada

    def generate():
        while True:
            frame = get_frame()
            if frame is None:
                continue

            noisy_frame = add_speckle_noise(frame, var)
            encoded = encode_frame(noisy_frame)
            if encoded:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

