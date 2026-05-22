from ultralytics import YOLO

DATASET = r'C:\Users\Chenz\Desktop\flower_photos\dataset'

model = YOLO('yolo11n-cls.pt')

results = model.train(
    data=DATASET,
    epochs=50,
    imgsz=224,
    batch=16,
    device='cuda',
    workers=4,
    name='flower_classifier',
    patience=10,
    lr0=0.001,
    seed=42,
)
