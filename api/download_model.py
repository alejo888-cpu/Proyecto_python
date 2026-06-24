"""
download_model.py — Script auxiliar para descargar/exportar el modelo ONNX.

Uso:
    python download_model.py

El script intenta, en orden:
  1. Exportar YOLOv8n a ONNX usando ultralytics (requiere ~6 MB de descarga
     desde GitHub; si la red lo bloquea, pasa al siguiente).
  2. Exportar MobileNetV2 a ONNX con pesos pre-entrenados de torchvision
     (similar red — puede fallar en entornos restringidos).
  3. Exportar MobileNetV2 a ONNX con pesos aleatorios (siempre funciona;
     las predicciones serán aleatorias hasta que reemplaces el archivo).
"""

import os, sys
import torch
import torchvision.models as models

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best.onnx")
DUMMY      = torch.randn(1, 3, 224, 224)

os.makedirs(MODEL_DIR, exist_ok=True)


def try_yolov8():
    print("[1/3] Intentando exportar YOLOv8n con ultralytics...")
    from ultralytics import YOLO
    import shutil, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        orig = os.getcwd()
        os.chdir(tmp)
        try:
            m = YOLO("yolov8n.pt")
            m.export(format="onnx", imgsz=640, simplify=True)
            shutil.move(os.path.join(tmp, "yolov8n.onnx"), MODEL_PATH)
            os.chdir(orig)
            return True
        except Exception as e:
            os.chdir(orig)
            print(f"   Falló: {e}")
            return False


def try_mobilenet_pretrained():
    print("[2/3] Intentando MobileNetV2 con pesos pre-entrenados...")
    try:
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        m.eval()
        torch.onnx.export(m, DUMMY, MODEL_PATH, opset_version=11,
                          input_names=["images"], output_names=["output"])
        return True
    except Exception as e:
        print(f"   Falló: {e}")
        return False


def try_mobilenet_random():
    print("[3/3] Exportando MobileNetV2 con pesos aleatorios (fallback)...")
    m = models.mobilenet_v2(weights=None)
    m.eval()
    torch.onnx.export(m, DUMMY, MODEL_PATH, opset_version=11,
                      input_names=["images"], output_names=["output"])
    return True


if __name__ == "__main__":
    for fn in [try_yolov8, try_mobilenet_pretrained, try_mobilenet_random]:
        if fn():
            size = os.path.getsize(MODEL_PATH)
            print(f"\n✅ Modelo guardado en: {MODEL_PATH}  ({size:,} bytes)")
            sys.exit(0)

    print("\n❌ No se pudo generar el modelo.")
    sys.exit(1)
