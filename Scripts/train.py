from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/home/oliver18/Documents/Personal_Projects/Cats_Classification/Data/Data_labeled/data.yaml", epochs=10, imgsz=640)