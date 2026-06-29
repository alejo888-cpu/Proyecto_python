"""
detector.py — Motor de inferencia ONNX para detección de EPP (YOLOv8).

El archivo api/model/best.onnx es un YOLOv8s fine-tuneado para EPP
(Construction Site Safety Dataset, Roboflow):
  - Input:  "images"  -> [1, 3, 640, 640]  (float32, normalizado 0-1, RGB)
  - Output: "output0"  -> [1, 14, 8400]
      14 = 4 coords de bounding box (cx, cy, w, h) + 10 clases
      8400 = número de celdas/anchors candidatas

Clases del modelo (10): Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
Person, Safety Cone, Safety Vest, machinery, vehicle.

Nota: este modelo NO incluye "gafas de seguridad" ni "botas de seguridad"
como clases — el dataset de origen (Construction Site Safety, Roboflow)
no las contempla. Cubre casco (Hardhat) y chaleco (Safety Vest), que son
los dos EPP más comunes de detectar visualmente y los que tienen mejor
soporte de datasets públicos.
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best.onnx")

INPUT_SIZE = 640          # YOLOv8 usa entradas cuadradas de 640x640
CONF_THRESHOLD = 0.45     # confianza mínima por caja antes de NMS.
                          # Subido de 0.25 a 0.45: con imágenes fuera del
                          # dominio de entrenamiento (objetos EPP aislados
                          # sobre fondo blanco, sin persona) el modelo
                          # generaba falsos positivos de baja confianza
                          # (~25-35%). 0.45 filtra ese ruido sin perder
                          # detecciones reales (que en escenas reales de
                          # obra suelen superar 70-90%).
IOU_THRESHOLD = 0.45       # umbral de solapamiento para NMS

# Clases del modelo de EPP (10), en el orden de entrenamiento (data.yaml).
PPE_CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]

# Traducción amigable para mostrar en la interfaz en español.
CLASS_LABELS_ES = {
    "Hardhat": "Casco de seguridad",
    "Mask": "Mascarilla",
    "NO-Hardhat": "Sin casco",
    "NO-Mask": "Sin mascarilla",
    "NO-Safety Vest": "Sin chaleco",
    "Person": "Persona",
    "Safety Cone": "Cono de seguridad",
    "Safety Vest": "Chaleco de seguridad",
    "machinery": "Maquinaria",
    "vehicle": "Vehículo",
}


# ---------------------------------------------------------------------------
# Helpers de inicialización
# ---------------------------------------------------------------------------

def _ensure_model() -> None:
    """Verifica que el archivo ONNX existe y tiene contenido."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        raise FileNotFoundError(
            f"Modelo ONNX no encontrado o vacío en: {MODEL_PATH}\n"
            "Coloca el archivo best.onnx (modelo de EPP) en la carpeta "
            "api/model/."
        )


# ---------------------------------------------------------------------------
# Preprocesamiento (letterbox + normalización 0-1, formato YOLOv8)
# ---------------------------------------------------------------------------

def _letterbox(img: Image.Image, size: int = INPUT_SIZE):
    """
    Redimensiona la imagen manteniendo el aspect ratio y rellena con
    padding gris (114,114,114) hasta llegar a size x size, igual que
    hace Ultralytics internamente. Devuelve la imagen letterboxed junto
    con el factor de escala y el padding aplicado (para poder reescalar
    las cajas de vuelta a las coordenadas de la imagen original).
    """
    orig_w, orig_h = img.size
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return canvas, scale, pad_x, pad_y


def preprocess(image_path: str):
    """
    Lee una imagen, aplica letterbox a 640x640 y devuelve:
      - tensor [1, 3, 640, 640] float32 normalizado 0-1 (RGB, sin restar media)
      - metadata (scale, pad_x, pad_y, orig_w, orig_h) para reescalar cajas
    """
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    orig_w, orig_h = img.size

    canvas, scale, pad_x, pad_y = _letterbox(img, INPUT_SIZE)

    arr = np.array(canvas, dtype=np.float32) / 255.0   # [H, W, 3] -> [0,1]
    arr = arr.transpose(2, 0, 1)                          # [3, H, W]
    arr = np.expand_dims(arr, axis=0)                     # [1, 3, H, W]

    meta = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "orig_w": orig_w,
        "orig_h": orig_h,
    }

    # Diagnóstico: comparar imagen de cámara vs galería
    print(f"[preprocess] orig: {orig_w}x{orig_h} | scale={scale:.4f} | pad=({pad_x},{pad_y})")
    print(f"[preprocess] tensor shape={arr.shape} | dtype={arr.dtype} | min={arr.min():.4f} | max={arr.max():.4f}")

    return arr, meta


# ---------------------------------------------------------------------------
# Postprocesamiento: decodificar salida YOLOv8 + NMS
# ---------------------------------------------------------------------------

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """NMS clásico. boxes en formato [x1, y1, x2, y2]. Devuelve índices a conservar."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_threshold]

    return keep


def _decode_output(output: np.ndarray, meta: dict, conf_threshold: float):
    """
    output: array [14, 8400] (ya sin la dimensión de batch).
    Devuelve lista de dicts: {"clase": str, "clase_es": str, "confianza": float (0-100), "box": [x1,y1,x2,y2]}
    con las cajas ya reescaladas a las coordenadas de la imagen original.

    IMPORTANTE: las clases de este modelo NO son mutuamente excluyentes.
    Una misma región de la imagen puede generar a la vez una detección de
    "Person" y, superpuesta, una de "Hardhat" o "NO-Safety Vest" — son
    objetos distintos, no una clasificación de una sola etiqueta por caja.
    Por eso NO se puede hacer un argmax() global por caja (eso descartaría
    cualquier clase EPP cuyo score, aunque sea alto, quede por debajo del
    score de "Person" en esa misma celda). En su lugar, cada clase se
    evalúa de forma independiente contra el umbral, y el NMS se aplica
    por clase (estándar en detección multi-objeto tipo YOLO).
    """
    n_classes = output.shape[0] - 4

    print(f"[DIAGNOSTICO] output.shape = {output.shape} | n_classes calculado = {n_classes}")

    # output[0:4]  -> cx, cy, w, h (en coordenadas de la imagen 640x640)
    # output[4:]   -> score por cada clase del modelo
    boxes_raw = output[0:4, :].T          # [8400, 4]
    scores_raw = output[4:, :].T          # [8400, n_classes]

    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)   # [8400, 4]

    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    results = []

    # Una clase a la vez: filtrar por umbral, NMS dentro de esa clase,
    # y solo entonces pasar a la siguiente. Así una caja puede aportar
    # detecciones a varias clases distintas si supera el umbral en cada una.
    for cls_id in range(n_classes):
        cls_scores = scores_raw[:, cls_id]
        cls_mask = cls_scores >= conf_threshold

        if not np.any(cls_mask):
            continue

        cls_boxes = boxes_xyxy[cls_mask]
        cls_confidences = cls_scores[cls_mask]

        keep = _nms(cls_boxes, cls_confidences, IOU_THRESHOLD)

        clase = PPE_CLASSES[cls_id] if cls_id < len(PPE_CLASSES) else f"clase_{cls_id}"
        clase_es = CLASS_LABELS_ES.get(clase, clase)

        for idx in keep:
            bx1, by1, bx2, by2 = cls_boxes[idx]

            # Quitar el padding del letterbox y reescalar a la imagen original.
            bx1 = (bx1 - pad_x) / scale
            by1 = (by1 - pad_y) / scale
            bx2 = (bx2 - pad_x) / scale
            by2 = (by2 - pad_y) / scale

            # Recortar a los límites de la imagen original.
            bx1 = max(0.0, min(bx1, orig_w))
            by1 = max(0.0, min(by1, orig_h))
            bx2 = max(0.0, min(bx2, orig_w))
            by2 = max(0.0, min(by2, orig_h))

            results.append({
                "clase": clase,
                "clase_es": clase_es,
                "confianza": round(float(cls_confidences[idx]) * 100, 2),
                "box": [round(float(bx1), 1), round(float(by1), 1), round(float(bx2), 1), round(float(by2), 1)],
            })

    # Ordenar por confianza descendente
    results.sort(key=lambda r: r["confianza"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Clase Detector
# ---------------------------------------------------------------------------

class Detector:
    """Encapsula la sesión ONNX y expone el método predict()."""

    def __init__(self):
        _ensure_model()
        self.labels = PPE_CLASSES

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(MODEL_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[detector] Modelo cargado: {MODEL_PATH}")
        print(f"[detector] Input: {self.input_name}")
        print(f"[detector] Clases: {self.labels}")

    def predict(self, image_path: str, top_k: int = 10) -> list[dict]:
        """
        Ejecuta la inferencia sobre la imagen en image_path.

        Devuelve una lista de dicts ordenada por confianza descendente,
        máximo top_k elementos:
          [{"clase": str, "clase_es": str, "confianza": float (0-100), "box": [x1,y1,x2,y2]}, ...]
        """
        tensor, meta = preprocess(image_path)
        outputs = self.session.run(None, {self.input_name: tensor})
        output0 = outputs[0][0]   # quitar dim de batch -> [n_classes+4, 8400]

        # Log de diagnóstico: mostrar TODAS las detecciones sin filtro de umbral
        results_raw = _decode_output(output0, meta, 0.01)  # umbral mínimo para ver todo
        print(f"[predict] Detecciones crudas (umbral 1%): {len(results_raw)}")
        for r in results_raw[:15]:
            print(f"  {r['clase']:<20} {r['confianza']:6.2f}%")

        results = _decode_output(output0, meta, CONF_THRESHOLD)
        print(f"[predict] Detecciones finales (umbral {CONF_THRESHOLD}%): {len(results)}")
        return results[:top_k]


# ---------------------------------------------------------------------------
# Instancia global (lazy) — se inicializa la primera vez que se usa
# ---------------------------------------------------------------------------
_detector_instance: Detector | None = None


def get_detector() -> Detector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = Detector()
    return _detector_instance


# ---------------------------------------------------------------------------
# NOTA SOBRE EPP
# ---------------------------------------------------------------------------
# Este modelo (YOLOv8s, fine-tuneado sobre el "Construction Site Safety
# Dataset" de Roboflow Universe) SÍ detecta EPP real: casco (Hardhat) y
# chaleco (Safety Vest), además de sus contrapartes de incumplimiento
# (NO-Hardhat, NO-Safety Vest), personas, conos de seguridad, maquinaria
# y vehículos. NO incluye gafas de seguridad ni botas como clases — el
# dataset de origen no las contempla. Si se necesitan esas dos clases
# específicas, habría que conseguir o anotar un dataset que las incluya
# y re-entrenar; el resto del pipeline (preprocesamiento, NMS) seguiría
# funcionando igual.