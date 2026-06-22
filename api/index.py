from flask import Flask, request, jsonify, render_template
import os

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    file = request.files["file"]

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    print("Guardado en:", filepath)  # Para verificar

    return jsonify({
        "status": "ok",
        "filename": file.filename,
        "ruta": filepath
    })


if __name__ == "__main__":
    app.run(debug=True)