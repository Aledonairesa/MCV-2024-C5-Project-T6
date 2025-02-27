from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.predict("/data/users/mireia/MCV/C5/KITTI-MOTS/testing/image_02/0000", save=True, imgsz=320, conf=0.5)