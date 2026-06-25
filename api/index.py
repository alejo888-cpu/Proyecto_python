"""
index.py — Servidor Flask para detección/clasificación de imágenes con ONNX.
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import RequestEntityTooLarge
from detector import get_detector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
CONF_THRESHOLD = 45.0  # % mínimo de confianza para considerar una detección válida.
                       # Sincronizado con CONF_THRESHOLD de detector.py (0.45).
                       # detector.py ya filtra internamente antes del NMS;
                       # este umbral en index.py es una segunda capa de
                       # seguridad en la misma escala (0-100).

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    return jsonify({
        "error": "ARCHIVO_DEMASIADO_GRANDE",
        "mensaje": "La imagen supera el límite de 16 MB."
    }), 413


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        det = get_detector()

        return jsonify({
            "status": "ok",
            "modelo": "cargado",
            "etiquetas": len(det.labels)
        })

    except Exception as exc:

        return jsonify({
            "status": "error",
            "detalle": str(exc)
        }), 503


@app.route("/detect", methods=["POST"])
def detect():

    # Validar archivo
    if "file" not in request.files:
        return jsonify({
            "error": "SIN_ARCHIVO",
            "mensaje": "No se recibió ninguna imagen."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "NOMBRE_VACIO",
            "mensaje": "El archivo no tiene nombre."
        }), 400

    # Validar extensión
    if not allowed_file(file.filename):

        ext = (
            file.filename.rsplit(".", 1)[-1]
            if "." in file.filename
            else "desconocido"
        )

        return jsonify({
            "error": "FORMATO_NO_SOPORTADO",
            "mensaje": f"Formato .{ext} no soportado."
        }), 415

    # Guardar archivo
    ext = file.filename.rsplit(".", 1)[1].lower()

    safe_name = f"{uuid.uuid4().hex}.{ext}"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        safe_name
    )

    try:
        file.save(filepath)

    except Exception as exc:

        print("\n========== ERROR GUARDANDO ARCHIVO ==========")
        print(type(exc).__name__)
        print(exc)
        print("============================================\n")

        return jsonify({
            "error": "ERROR_AL_GUARDAR",
            "mensaje": str(exc)
        }), 500

    # Inferencia
    try:

        detector = get_detector()

        resultados = detector.predict(
            filepath,
            top_k=5
        )

    except FileNotFoundError as exc:

        print("\n========== MODELO NO ENCONTRADO ==========")
        print(exc)
        print("=========================================\n")

        return jsonify({
            "error": "MODELO_NO_ENCONTRADO",
            "mensaje": str(exc)
        }), 503

    except Exception as exc:

        print("\n========== ERROR DETECTOR ==========")
        print(type(exc).__name__)
        print(exc)
        print("===================================\n")

        return jsonify({
            "error": "ERROR_INFERENCIA",
            "mensaje": str(exc)
        }), 500

    # Filtrar resultados
    detectados = [
        r
        for r in resultados
        if r["confianza"] >= CONF_THRESHOLD
    ]

    if not detectados:

        return jsonify({
            "status": "sin_detecciones",
            "mensaje": "No se detectaron objetos con suficiente confianza.",
            "resultados": [],
            "archivo": safe_name
        })

    # Respuesta OK
    return jsonify({
        "status": "ok",
        "archivo": safe_name,
        "top_k": len(detectados),
        "resultados": detectados
    })


if __name__ == "__main__":

    print("[index] Precargando modelo...")

    get_detector()

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )