# Author: vlarobbyk (modificado por Diego)
# Version: 1.1
# Date: 2025-05-01
# Description: Detección de movimiento usando Adaptive Background Subtraction (MOG2) en Flask

from flask import Flask, render_template, Response
from io import BytesIO
import cv2
import numpy as np
import requests

app = Flask(__name__)

# Configuración de la cámara
_URL = 'http://192.168.18.248' 
_PORT = '81'
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL, SEP, _PORT, _ST])

# Crea el objeto Background Subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

def video_capture():
    try:
        res = requests.get(stream_url, stream=True)
        for chunk in res.iter_content(chunk_size=100000):
            if len(chunk) > 100:
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                # Convertir a escala de grises
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Aplicar la substracción de fondo
                fgMask = backSub.apply(gray)

                # Encontrar contornos del movimiento
                contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Evita pequeños ruidos
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Codificar el frame procesado
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue

                # Enviar frame al navegador
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    except Exception as e:
        print(f"Error capturando vídeo: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    return Response(video_capture(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)

