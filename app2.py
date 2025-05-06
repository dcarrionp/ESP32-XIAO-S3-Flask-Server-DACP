# Author: Diego Andrés Carrión Portilla
# Description: Aplicación Flask para operaciones morfológicas en imágenes médicas.

from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# ========== CONFIGURACIÓN ==========
IMAGES = {
    "nih": "/home/diego/Vision/ESP32-XIAO-S3-Flask-Server-DACP/static/images/imagen1",
    "pneumonia": "/home/diego/Vision/ESP32-XIAO-S3-Flask-Server-DACP/static/images/imagen2",
    "covid": "/home/diego/Vision/ESP32-XIAO-S3-Flask-Server-DACP/static/images/imagen3"
}

KERNEL_SIZE = 37
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))

# ========== FUNCIONES ==========
def load_image(path):
    """Carga una imagen en escala de grises."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return img

def encode_image(image):
    """Codifica una imagen para transmitirla como stream."""
    success, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes() if success else None

def apply_morph_operation(image, operation):
    """Aplica una operación morfológica."""
    if operation == "original":
        return image
    elif operation == "eroded":
        return cv2.erode(image, KERNEL, iterations=1)
    elif operation == "dilated":
        return cv2.dilate(image, KERNEL, iterations=1)
    elif operation == "tophat":
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, KERNEL)
    elif operation == "blackhat":
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, KERNEL)
    elif operation == "enhanced":
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, KERNEL)
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, KERNEL)
        enhanced = cv2.add(image, cv2.subtract(tophat, blackhat))
        return enhanced
    else:
        raise ValueError(f"Operación no soportada: {operation}")

def generate_stream(image_name, operation):
    """Genera un stream de la imagen procesada."""
    img = load_image(IMAGES[image_name])
    processed_img = apply_morph_operation(img, operation)
    encoded = encode_image(processed_img)
    if encoded:
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

# ========== RUTAS FLASK ==========

@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/<dataset>/<operation>")
def show_operation(dataset, operation):
    """Genera el video de la imagen solicitada con la operación seleccionada."""
    if dataset not in IMAGES:
        return "Dataset no encontrado", 404
    try:
        return Response(generate_stream(dataset, operation), mimetype='multipart/x-mixed-replace; boundary=frame')
    except ValueError as e:
        return str(e), 400

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

