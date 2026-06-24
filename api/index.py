"""
index.py — Servidor Flask para detección/clasificación de imágenes con ONNX.
"""

import os, uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from detector import get_detector

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT   = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
CONF_THRESHOLD = 2.0   # % mínimo de confianza para considerar una detección válida

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    return jsonify({
        "error": "ARCHIVO_DEMASIADO_GRANDE",
        "mensaje": "La imagen supera el límite de 16 MB. Usa una imagen más pequeña."
    }), 413


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        det = get_detector()
        return jsonify({"status": "ok", "modelo": "cargado", "etiquetas": len(det.labels)})
    except Exception as exc:
        return jsonify({"status": "error", "detalle": str(exc)}), 503


@app.route("/detect", methods=["POST"])
def detect():
    # ── 1. Validar presencia del archivo ────────────────────────────────
    if "file" not in request.files:
        return jsonify({
            "error": "SIN_ARCHIVO",
            "mensaje": "No se recibió ninguna imagen. Selecciona o captura una foto primero."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "NOMBRE_VACIO",
            "mensaje": "El archivo no tiene nombre. Intenta seleccionarlo de nuevo."
        }), 400

    # ── 2. Validar formato ───────────────────────────────────────────────
    if not allowed_file(file.filename):
        ext = file.filename.rsplit(".", 1)[-1] if "." in file.filename else "desconocido"
        return jsonify({
            "error": "FORMATO_NO_SOPORTADO",
            "mensaje": f"El formato «.{ext}» no está soportado. Usa: JPG, PNG, WEBP, BMP o GIF."
        }), 415

    # ── 3. Guardar imagen ────────────────────────────────────────────────
    ext       = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    filepath  = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        file.save(filepath)
    except Exception as exc:
        return jsonify({
            "error": "ERROR_AL_GUARDAR",
            "mensaje": "No se pudo guardar la imagen en el servidor. Intenta de nuevo."
        }), 500

    # ── 4. Inferencia ────────────────────────────────────────────────────
    try:
        detector   = get_detector()
        resultados = detector.predict(filepath, top_k=5)
    except FileNotFoundError as exc:
        return jsonify({
            "error": "MODELO_NO_ENCONTRADO",
            "mensaje": "El modelo ONNX no está disponible. Ejecuta download_model.py y reinicia el servidor."
        }), 503
    except Exception as exc:
        return jsonify({
            "error": "ERROR_INFERENCIA",
            "mensaje": "El modelo no pudo procesar la imagen. Prueba con una foto más nítida o en otro formato."
        }), 500

    # ── 5. Umbral de confianza ───────────────────────────────────────────
    detectados = [r for r in resultados if r["confianza"] >= CONF_THRESHOLD]

    if not detectados:
        return jsonify({
            "status":    "sin_detecciones",
            "mensaje":   "No se detectaron objetos con suficiente confianza. "
                         "Intenta con una imagen más clara, mejor iluminada o con un objeto más visible.",
            "resultados": [],
            "archivo":   safe_name,
        })

    # ── 6. Respuesta exitosa ─────────────────────────────────────────────
    return jsonify({
        "status":     "ok",
        "archivo":    safe_name,
        "top_k":      len(detectados),
        "resultados": detectados,
    })


if __name__ == "__main__":
    print("[index] Precargando modelo...")
    get_detector()
    app.run(debug=True, host="0.0.0.0", port=5000)
