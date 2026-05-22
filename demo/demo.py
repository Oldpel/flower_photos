import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\Chenz\Desktop\flower_photos\runs\classify\flower_classifier-2\weights\best.pt"
model = YOLO(MODEL_PATH)

CLASS_CN = {
    'daisy': '雏菊',
    'dandelion': '蒲公英',
    'roses': '玫瑰',
    'sunflowers': '向日葵',
    'tulips': '郁金香',
}

IMG_SIZE = 400


class FlowerClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('花卉分类器')
        self.root.geometry('520x620')
        self.root.resizable(False, False)

        # --- 顶部标题 ---
        title = ttk.Label(root, text='花卉分类器', font=('Microsoft YaHei', 18, 'bold'))
        title.pack(pady=(16, 8))

        # --- 图片预览区 ---
        frame = ttk.Frame(root, width=IMG_SIZE, height=IMG_SIZE, relief='solid')
        frame.pack(pady=4)
        frame.pack_propagate(False)

        self.img_label = ttk.Label(frame, text='请选择一张图片', font=('Microsoft YaHei', 12), foreground='gray')
        self.img_label.pack(expand=True)

        # --- 选择按钮 ---
        self.btn = ttk.Button(root, text='选择图片', command=self.select_image)
        self.btn.pack(pady=(12, 8))

        # --- 进度条 ---
        self.progress = ttk.Progressbar(root, mode='indeterminate', length=300)

        # --- 结果展示 ---
        self.result_frame = ttk.Frame(root)
        self.result_frame.pack(pady=4)

        self.flower_label = ttk.Label(
            self.result_frame, text='', font=('Microsoft YaHei', 16, 'bold'), foreground='#2e7d32'
        )
        self.flower_label.pack()

        self.conf_label = ttk.Label(
            self.result_frame, text='', font=('Microsoft YaHei', 11), foreground='#555'
        )
        self.conf_label.pack()

        # 状态
        self.photo = None

    def select_image(self):
        path = filedialog.askopenfilename(
            title='选择花卉图片',
            filetypes=[('图片文件', '*.jpg *.jpeg *.png *.bmp'), ('所有文件', '*.*')],
        )
        if not path:
            return

        self.show_preview(path)
        self.classify(path)

    def show_preview(self, path):
        img = Image.open(path).convert('RGB')
        img.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)

        self.img_label.configure(image=self.photo, text='', foreground='gray')
        self.img_label.image = self.photo  # 防止被 GC

    def classify(self, path):
        self.btn.configure(state='disabled', text='分析中…')
        self.progress.pack(pady=(0, 8))
        self.progress.start(10)
        self.flower_label.configure(text='')
        self.conf_label.configure(text='')
        self.root.update()

        try:
            results = model.predict(path, imgsz=224, device='cpu', verbose=False)
            result = results[0]
            top_idx = result.probs.top1
            top_name = result.names[top_idx]
            top_conf = result.probs.top1conf.item()

            cn_name = CLASS_CN.get(top_name, top_name)
            self.flower_label.configure(text=f'{cn_name} ({top_name})')
            self.conf_label.configure(text=f'置信度：{top_conf:.2%}')
        except Exception as e:
            self.flower_label.configure(text='分析失败', foreground='red')
            self.conf_label.configure(text=str(e))
        finally:
            self.progress.stop()
            self.progress.pack_forget()
            self.btn.configure(state='normal', text='选择图片')


if __name__ == '__main__':
    root = tk.Tk()
    app = FlowerClassifierApp(root)
    root.mainloop()
