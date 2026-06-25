"""
Script de diagnóstico — pega esto en tu carpeta api/ y ejecútalo así:
    python diagnostico.py "ruta/a/Casco2.jpg"

Muestra, para CADA clase, el score MÁXIMO encontrado en toda la imagen,
sin aplicar ningún umbral. Así vemos si Hardhat tiene 0%, o tiene algo
de actividad que simplemente no llega al 45%.
"""
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, ".")
from detector import preprocess, MODEL_PATH, PPE_CLASSES

if len(sys.argv) < 2:
    print("Uso: python diagnostico.py ruta_a_imagen.jpg")
    sys.exit(1)

image_path = sys.argv[1]

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

tensor, meta = preprocess(image_path)
outputs = session.run(None, {input_name: tensor})
output0 = outputs[0][0]   # [14, 8400]

scores = output0[4:, :]   # [10, 8400]

print(f"\nImagen: {image_path}")
print(f"Tamaño original: {meta['orig_w']}x{meta['orig_h']}\n")
print(f"{'Clase':<20} {'Score máx (%)':<15} {'En qué caja (índice)'}")
print("-" * 55)
for i, clase in enumerate(PPE_CLASSES):
    cls_scores = scores[i, :]
    max_score = cls_scores.max()
    max_idx = cls_scores.argmax()
    print(f"{clase:<20} {max_score*100:>6.2f}%        caja #{max_idx}")