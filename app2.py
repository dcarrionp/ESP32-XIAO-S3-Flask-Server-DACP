# Author: Diego Andrés Carrión Portilla
# Description: Aplicación Flask para aplicar operaciones morfológicas en imágenes médicas

from flask import Flask, render_template, Response
import cv2
import numpy as np
import os

app = Flask(__name__)

# ========== CONFIGURACIÓN ==========
# Define las rutas a las imágenes seleccionadas
IMAGES = {
    "nih": "/home/diego/Vision/ESP32-XIAO-S3-Flask-Server-DACP/static/images/imagen1",
    "pneumonia": "/home/diego/Vision/ESP32-XIAO-S3-Flask-Server-DACP/static/images/imagen2",
    "covid": "/home/diego/Vision/ESP32-XIAO-S3-Flask-Server-DACP/static/images/imagen3"
}

# Kernel para operaciones morfológicas
KERNEL_SIZE = 37  # Tamaño grande como sugiere la práctica
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))

# ========== FUNCIONES AUXILIARES ==========

def load_image(path):
    """Carga una imagen en escala de grises."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return img

def encode_image(image):
    """Codifica la imagen para ser enviada como streaming."""
    success, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes() if success else None

# ========== FLASK ROUTES ==========

@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/nih_original")
def nih_original():
    img = load_image(IMAGES["nih"])
    encoded = encode_image(img)
    return Response((b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/nih_eroded")
def nih_eroded():
    img = load_image(IMAGES["nih"])
    eroded = cv2.erode(img, KERNEL, iterations=1)
    encoded = encode_image(eroded)
    return Response((b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/pneumonia_original")
def pneumonia_original():
    img = load_image(IMAGES["pneumonia"])
    encoded = encode_image(img)
    return Response((b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/pneumonia_eroded")
def pneumonia_eroded():
    img = load_image(IMAGES["pneumonia"])
    eroded = cv2.erode(img, KERNEL, iterations=1)
    encoded = encode_image(eroded)
    return Response((b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/covid_original")
def covid_original():
    img = load_image(IMAGES["covid"])
    encoded = encode_image(img)
    return Response((b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/covid_eroded")
def covid_eroded():
    img = load_image(IMAGES["covid"])
    eroded = cv2.erode(img, KERNEL, iterations=1)
    encoded = encode_image(eroded)
    return Response((b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ========== MAIN ==========

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

