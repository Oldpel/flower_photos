from ultralytics import YOLO

MODEL_PATH = r"C:\Users\Chenz\Desktop\flower_photos\runs\classify\flower_classifier-2\weights\best.pt"
DATASET = r"C:\Users\Chenz\Pictures\Screenshots\屏幕截图 2026-05-22 170103.png"
model = YOLO(MODEL_PATH)

results = model.predict(DATASET, imgsz=224, device='cpu')

result = results[0]
top_idx = result.probs.top1
top_name = result.names[top_idx]
top_conf = result.probs.top1conf.item()
print(f'{top_name}，置信度 {top_conf:.6%}')
