from ultralytics import YOLO
import pickle

data_path = "/data/users/mireia/MCV/C5/YOLO-KITTI-MOTS/all_frames"
project = 'pred_outputs'
name = 'pred_off-shelf'

model = YOLO("yolo11x.pt")

preds = model.predict(
    source = data_path,
    conf = 0.25,  # Minimum confidence threshold for detections
    iou = 0.65,  # IoU threshold for NMS
    imgsz = 640,
    half = True,
    device = 'cuda:7',
    batch = 16,
    max_det = 300,
    visualize = False,
    augment = False,
    agnostic_nms = False,
    classes = [0,2],  # Labels to detect on COCO
    embed = False,
    project = project,
    name = name,
    save = False
)

# Save preds to pkl file
with open(f"{project}/{name}.pkl", "wb") as f:
    pickle.dump(preds, f)
