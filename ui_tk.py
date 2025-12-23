import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import torch
from torchvision import transforms
import os

classes = ['Esox_lucius', 'Cyprinus_carpio', 'Carcharodon_carcharias',
           'Silurus_glanis', 'Delphinapterus_leucas']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(classes))
model.load_state_dict(torch.load('best_fish_classifier.pth', map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FishClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Классификация Рыб")
        self.geometry("600x400")
        self.img_label = None
        self.result_label = None
        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self)
        frame.pack(fill="both", expand=True)

        open_button = tk.Button(frame, text="Открыть изображение", command=self.open_image)
        open_button.pack(side="top", pady=(20, 10))

        self.img_label = tk.Label(frame)
        self.img_label.pack(side="top", padx=20, pady=20)

        self.result_label = tk.Label(frame, font=("Arial", 16))
        self.result_label.pack(side="bottom", pady=(20, 10))

        predict_button = tk.Button(frame, text="Определить вид рыбы", command=self.predict)
        predict_button.pack(side="bottom", pady=(10, 20))

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ("*.jpg", "*.jpeg", "*.png"))])
        if file_path:
            try:
                self.result_label.config(text=f"", fg="green")
                img = Image.open(file_path)
                resized_img = img.resize((300, 300), Image.LANCZOS)
                photo = ImageTk.PhotoImage(resized_img)
                self.img_label.config(image=photo)
                self.img_label.image = photo
                self.img = Image.open(file_path)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Невозможно открыть выбранное изображение.\n{e}")

    def predict(self):
        if hasattr(self, 'img'):
            transformed_img = transform(self.img).unsqueeze(0).to(device)
            output = model(transformed_img)
            prediction = torch.argmax(output, dim=1)[0].item()
            predicted_class = classes[prediction]
            self.result_label.config(text=f"Предсказанная рыба: {predicted_class}", fg="green")
        else:
            messagebox.showwarning("Внимание", "Сначала выберите изображение!")

if __name__ == "__main__":
    app = FishClassifierApp()
    app.mainloop()