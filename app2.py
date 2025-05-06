# Author: Diego Andrés Carrión Portilla
# Description: Aplicación Flask para operaciones morfológicas en imágenes médicas con 3 tamaños de máscara

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

# Tamaños de máscara
KERNEL_SIZES = [15, 37, 51]

# ========== FUNCIONES ==========
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return img

def encode_image(image):
    success, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes() if success else None

def apply_operation(image, operation, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if operation == "erode":
        return cv2.erode(image, kernel, iterations=1)
    elif operation == "dilate":
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == "tophat":
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    elif operation == "blackhat":
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    elif operation == "enhanced":
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        enhanced = cv2.add(image, cv2.subtract(tophat, blackhat))
        return enhanced
    else:
        return image

def generate_stream(image_name, operation, kernel_size):
    img = load_image(IMAGES[image_name])
    processed = apply_operation(img, operation, kernel_size)
    encoded = encode_image(processed)
    if encoded:
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded + b'\r\n')

# ========== RUTAS ==========
@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/<image_name>/<operation>/<int:kernel_size>")
def stream_operation(image_name, operation, kernel_size):
    return Response(generate_stream(image_name, operation, kernel_size),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

