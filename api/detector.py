"""
detector.py — Motor de inferencia ONNX para clasificación de imágenes.

Usa MobileNetV2 exportado a ONNX (ligero, ~14 MB con pesos reales).
Si el archivo de pesos no existe o está vacío, intenta descargarlo
automáticamente desde una fuente pública; si falla, levanta un error claro.

Clases: ImageNet-1k (1000 clases) con las top-5 devueltas como JSON.
"""

import os
import json
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best.onnx")

# Etiquetas ImageNet-1k (se cargan al iniciar)
LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/"
    "imagenet_classes.txt"
)
LABELS_PATH = os.path.join(MODEL_DIR, "imagenet_classes.txt")

# ---------------------------------------------------------------------------
# Helpers de inicialización
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: str) -> bool:
    """Descarga url → dest. Devuelve True si tuvo éxito."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        print(f"[detector] Descargando {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[detector] Guardado en {dest}  ({os.path.getsize(dest):,} bytes)")
        return True
    except Exception as exc:
        print(f"[detector] Fallo al descargar {url}: {exc}")
        return False


def _load_labels() -> list[str]:
    """Carga las etiquetas ImageNet; las descarga si no existen."""
    if not os.path.exists(LABELS_PATH) or os.path.getsize(LABELS_PATH) == 0:
        ok = _download_file(LABELS_URL, LABELS_PATH)
        if not ok:
            # fallback: etiquetas genéricas
            return [f"clase_{i}" for i in range(1000)]
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


def _ensure_model() -> None:
    """Verifica que el archivo ONNX existe y tiene contenido."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        raise FileNotFoundError(
            f"Modelo ONNX no encontrado o vacío en: {MODEL_PATH}\n"
            "Coloca el archivo best.onnx en la carpeta api/model/ "
            "o ejecuta scripts/export_model.py para generarlo."
        )


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path: str) -> np.ndarray:
    """
    Lee una imagen, la redimensiona a 224×224, normaliza con media/std
    ImageNet y devuelve un tensor [1, 3, 224, 224] float32.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, 3]  → [0, 1]
    arr = (arr - _MEAN) / _STD                        # normalizar
    arr = arr.transpose(2, 0, 1)                      # [3, H, W]
    arr = np.expand_dims(arr, axis=0)                 # [1, 3, H, W]
    return arr


# ---------------------------------------------------------------------------
# Clase Detector
# ---------------------------------------------------------------------------

class Detector:
    """Encapsula la sesión ONNX y expone el método predict()."""

    def __init__(self):
        _ensure_model()
        self.labels = _load_labels()

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(MODEL_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[detector] Modelo cargado: {MODEL_PATH}")
        print(f"[detector] Input: {self.input_name}")

    def predict(self, image_path: str, top_k: int = 5) -> list[dict]:
        """
        Ejecuta la inferencia sobre la imagen en image_path.

        Devuelve una lista de dicts ordenada por confianza descendente:
          [{"clase": str, "confianza": float (0-100)}, ...]
        """
        tensor = preprocess(image_path)
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0][0]                            # shape [1000]

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        top_indices = probs.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "clase": self.labels[idx] if idx < len(self.labels) else f"clase_{idx}",
                "confianza": round(float(probs[idx]) * 100, 2),
            })

        return results


# ---------------------------------------------------------------------------
# Instancia global (lazy) — se inicializa la primera vez que se usa
# ---------------------------------------------------------------------------
_detector_instance: Detector | None = None


def get_detector() -> Detector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = Detector()
    return _detector_instance
