import torch
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageOps
import numpy as np
from emnist_cnn import EMNISTCNN
# ascii letters
import string

class LetterClassifierApp:
    def __init__(self, root, model_path="src/8_ann_and_cnn/weights/model.weight"):
        self.model_path = model_path
        
        # Create the main window, canvas and buttons for the GUI
        self.root = root
        self.root.title("EMNIST Letter Classifier")
        
        self.canvas = Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()
        
        self.predict_button = tk.Button(root, text="Classify", command=self.classify)
        self.predict_button.pack()
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        
        self.label = tk.Label(root, text="Draw a letter and press 'Classify'")
        self.label.pack()
        
        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.paint)
        self.model = self.load_model()
        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageOps.invert(self.image).load()

    def load_model(self):
        """Load the pre-trained model from the specified path."""
        model = EMNISTCNN()
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device("cpu")))
        model.eval()
        return model

    def paint(self, event):
        """Handle mouse drag events to draw on the canvas."""
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="black", outline="black")
        for i in range(max(0, x-5), min(280, x+5)):
            for j in range(max(0, y-5), min(280, y+5)):
                self.image.putpixel((i // 10, j // 10), 255)

    def classify(self):
        """Classify the drawn letter using the pre-trained model."""
        resized_image = self.image.resize((28, 28))
        image_tensor = torch.tensor(np.array(resized_image) / 255.0).float().view(1, 1, 28, 28)
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1).item()
            self.label.config(text=f"Predicted Letter: {prediction} ({string.ascii_letters[prediction]})")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)

if __name__ == "__main__":
    root = tk.Tk()
    app = LetterClassifierApp(root)
    root.mainloop()