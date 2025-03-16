from ultralytics import YOLO

# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo11x-seg.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(
    data="kitti-mots.yaml",
    epochs=500,
    patience=100,
    batch=16,
    imgsz=640,
    save=True,
    save_period=-1,
    cache=False,
    device=0,
    workers=8,
    project='runs/train_outputs',
    name='train_finetune_noaug',
    pretrained=True,
    optimizer='auto',
    seed=0,
    deterministic=True,
    single_cls=False,
    multi_scale=False,
    cos_lr=False,
    close_mosaic=10,
    amp=True,
    freeze=None,  # TODO choose which layers to freeze
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    dropout=0.0,
    val=True,
    plots=True 
)