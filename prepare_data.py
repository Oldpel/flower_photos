import os
import shutil
from sklearn.model_selection import train_test_split

BASE = r'C:\Users\Chenz\Desktop\flower_photos'
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
DATASET = os.path.join(BASE, 'dataset')
TRAIN = os.path.join(DATASET, 'train')
VAL = os.path.join(DATASET, 'val')

for cls in CLASSES:
    os.makedirs(os.path.join(TRAIN, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL, cls), exist_ok=True)

for cls in CLASSES:
    src = os.path.join(BASE, cls)
    images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    for img in train_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(TRAIN, cls, img))
    for img in val_imgs:
        shutil.copy2(os.path.join(src, img), os.path.join(VAL, cls, img))
    print(f'{cls}: train={len(train_imgs)}, val={len(val_imgs)}')

print('Done.')
