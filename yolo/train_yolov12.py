from ultralytics import YOLO

# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12m.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="kitti-mots.yaml", epochs=100, imgsz=320)