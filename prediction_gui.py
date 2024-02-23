import tkinter as tk
import random
from PIL import ImageTk, Image
import csv
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models


class GUI:
    def __init__(self, master):
        import os
        # Delete previous CSV file
        if os.path.exists('image_emotion.csv'):
            os.remove('image_emotion.csv')
        self.master = master
        master.title("Manga Emotion Classifier")
        self.emotions = ['angry', 'happy', 'sad', 'shock']
        self.transformations = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.Grayscale(3),
            transforms.Lambda(lambda x: torchvision.transforms.functional.adjust_contrast(x, 2)),
            transforms.Lambda(lambda x: torchvision.transforms.functional.adjust_contrast(x, 2)),
            transforms.Lambda(lambda x: torchvision.transforms.functional.adjust_gamma(x, 2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.selected_emotion = tk.StringVar(master, value=self.emotions[0])
        self.emotion_menu = tk.OptionMenu(master, self.selected_emotion, *self.emotions)
        self.emotion_menu.pack()
        self.generate_button = tk.Button(master, text="Generate Image", command=self.generate_image)
        self.generate_button.pack()
        self.image_label = tk.Label(master)
        self.image_label.pack()
        self.emotion_label = tk.Label(master, text="")
        self.emotion_label.pack()
        self.dataset = ImageFolder(root="./human_faces", transform=self.transformations)
        self.header_written = False
    def generate_image(self):
        selected_emotion = self.selected_emotion.get()
        random_image_path, _ = random.choice(self.dataset.samples)
        image = Image.open(random_image_path)
        transformed_image = self.transformations(image)
        transformed_image = torch.unsqueeze(transformed_image, 0)
        my_model.eval()
        with torch.no_grad():
            prediction = my_model(transformed_image)

        try:
            predicted_emotion = self.emotions[torch.argmax(prediction, dim=1).item()]
        except IndexError:
            predicted_emotion = "N/A"
            self.emotion_label.configure(text="I don't know ):")
        else:
            self.emotion_label.configure(text=f"Predicted Emotion: {predicted_emotion}")

        predicted_emotion = self.emotions[torch.argmax(prediction, dim=1).item()]
        self.image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.image)
        self.emotion_label.configure(text=f"Predicted Emotion: {predicted_emotion}")
        with open('image_emotion.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if not self.header_written:
                writer.writerow(['User Selection', 'Predictions', 'Image Path', 'Feature 1', 'Feature 2', 'Feature 3'])
                self.header_written = True
            writer.writerow([selected_emotion, predicted_emotion, random_image_path, 0, 0, 0])

device = 'cpu'
my_model = models.resnet50(pretrained=False)
in_feats = my_model.fc.in_features
my_model.fc = nn.Linear(in_feats, 7)
my_model.load_state_dict(torch.load('my_model.pth'))


root = tk.Tk()
gui = GUI(root)
root.mainloop()
