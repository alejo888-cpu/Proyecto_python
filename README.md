# API de Detección de Imágenes — ONNX

Sistema de clasificación de imágenes con modelo ONNX ligero servido por Flask.

## Estructura

```
api/
├── api/
│   ├── index.py          ← Servidor Flask (endpoints)
│   ├── detector.py       ← Motor de inferencia ONNX
│   ├── download_model.py ← Script de descarga/exportación del modelo
│   └── model/
│       ├── best.onnx           ← Pesos del modelo (MobileNetV2 / YOLOv8n)
│       └── imagenet_classes.txt← Etiquetas ImageNet-1k
├── static/
│   ├── css/styles.css
│   ├── js/app.js
│   └── uploads/          ← Imágenes recibidas (generadas en runtime)
├── templates/
│   └── index.html
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
```

## Modelos ONNX (importante)

En este proyecto **ya se incluyen** los pesos en `api/model/best.onnx`. Por eso **no es necesario** ejecutar un script de descarga.

Para usar tu propio modelo entrenado:

- Coloca tu archivo `.onnx` en `api/model/best.onnx` (sobrescribiendo el existente) o ajusta el path en `api/detector.py`.


## Ejecución

```bash
cd api
python api/index.py
```

Servidor en `http://localhost:5000`

## Endpoints

### `GET /`

Devuelve la interfaz web.

### `GET /health`

```json
{ "status": "ok", "modelo": "cargado", "etiquetas": 1000 }
```

### `POST /detect`

Recibe una imagen (campo `file`) y devuelve las top-5 clases:

```json
{
  "status": "ok",
  "archivo": "a3f2...jpg",
  "top_k": 5,
  "resultados": [
    { "clase": "golden retriever", "confianza": 87.42 },
    { "clase": "Labrador retriever", "confianza": 6.31 },
    ...
  ]
}
```

## Notas técnicas

- **Preprocesamiento**: resize 224×224, normalización ImageNet (media/std).
- **Runtime**: `onnxruntime` CPU (sin dependencia de CUDA).
- **Carga lazy**: el modelo se instancia la primera vez que llega una petición (o al arrancar si se llama `get_detector()` en `__main__`).
- **Máximo upload**: 16 MB por imagen.
- **Formatos aceptados**: jpg, jpeg, png, webp, bmp, gif.
