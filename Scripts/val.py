from ultralytics import YOLO

# Cargar un modelo YOLOv8 entrenado
model = YOLO("runs/detect/train/weights/best.pt")

# Evaluar el modelo en el conjunto de validación especificado en el archivo YAML
results = model.val(data="Data/Data_labeled/data.yaml", imgsz=640)