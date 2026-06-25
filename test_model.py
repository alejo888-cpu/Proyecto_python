import onnxruntime as ort

session = ort.InferenceSession(
    "api/model/best.onnx",
    providers=["CPUExecutionProvider"]
)

print(session.get_modelmeta().custom_metadata_map)