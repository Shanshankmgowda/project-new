import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import nltk
from nltk.corpus import wordnet
import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas, Scrollbar, Frame
from PIL import Image, ImageTk
import winsound

# Download WordNet dataset
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Pre-Trained Model
print("Loading pre-trained MobileNetV2 model...")
model = MobileNetV2(weights="imagenet") 
print("Model loaded successfully!")

# Function to check if the label is a bird
def is_bird(label):
    bird_synset = wordnet.synset('bird.n.01')  # Base synset for birds
    synsets = wordnet.synsets(label)
    for synset in synsets:
        if bird_synset in synset.lowest_common_hypernyms(bird_synset):
            return True
    return False

# Function to predict species
def predict_species(image):
    image_resized = cv2.resize(image, (224, 224))
    input_data = np.expand_dims(image_resized, axis=0)
    input_data = preprocess_input(input_data)
    predictions = model.predict(input_data)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    filtered_predictions = [pred for pred in decoded_predictions if is_bird(pred[1])]
    return filtered_predictions

# Function to plot graphs
def plot_bar(predictions):
    labels = [label for (_, label, _) in predictions]
    scores = [score * 100 for (_, _, score) in predictions]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=scores, y=labels, palette='viridis')
    plt.xlabel('Confidence (%)')
    plt.title('Bar Plot of Predictions')
    plt.tight_layout()
    plt.show()

def plot_donut(predictions):
    labels = [label for (_, label, _) in predictions]
    scores = [score * 100 for (_, _, score) in predictions]
    plt.figure(figsize=(8, 6))
    plt.pie(scores, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors, wedgeprops={'width': 0.5})
    plt.title('Donut Chart of Predictions')
    plt.tight_layout()
    plt.show()

def plot_scatter(predictions):
    labels = [label for (_, label, _) in predictions]
    scores = [score * 100 for (_, _, score) in predictions]
    plt.figure(figsize=(8, 6))
    plt.scatter(labels, scores, color='red', s=100, edgecolors='black')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Confidence (%)')
    plt.title('Scatter Plot of Predictions')
    plt.tight_layout()
    plt.show()

# Function to open live camera
def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        predictions = predict_species(frame)
        if predictions:
            label, confidence = predictions[0][1], predictions[0][2] * 100
            cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            winsound.Beep(1000, 500)
        else:
            cv2.putText(frame, "No birds detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Bird Species Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to upload and detect birds in images
def upload_images():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.jfif")])
    if not file_paths:
        return
    
    for widget in image_frame.winfo_children():
        widget.destroy()
    
    for file_path in file_paths:
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions = predict_species(image_rgb)
        detected_birds = []
        
        for pred in predictions:
            label, confidence = pred[1], pred[2] * 100
            detected_birds.append(f"{label} ({confidence:.2f}%)")
        
        result_text = "Detected: " + ", ".join(detected_birds) if detected_birds else "No birds detected"
        display_image(file_path, result_text)
        
        if predictions:
            plot_bar(predictions)
            plot_donut(predictions)
            plot_scatter(predictions)

# Function to display images
def display_image(file_path, result_text):
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    
    img_label = Label(image_frame, image=img, text=result_text, compound="top", fg="white", bg="#2c3e50", font=("Arial", 12))
    img_label.image = img  # Keep reference to avoid garbage collection
    img_label.pack(pady=10)

# Initialize Tkinter Window
root = tk.Tk()
root.title("Bird Species Detection")
root.geometry("900x650")
root.configure(bg="#2c3e50")

# Buttons
upload_btn = Button(root, text="Upload Images", command=upload_images, font=("Arial", 14), bg="#2980b9", fg="white")
upload_btn.pack(pady=10)

camera_btn = Button(root, text="Open Camera", command=open_camera, font=("Arial", 14), bg="#2980b9", fg="white")
camera_btn.pack(pady=10)

exit_btn = Button(root, text="Exit", command=root.destroy, font=("Arial", 14), bg="#c0392b", fg="white")
exit_btn.pack(pady=10)

# Frame for displaying images
image_frame = Frame(root, bg="#2c3e50")
image_frame.pack()

footer_label = Label(root, text="College: Govt.Polytechnic M.H Halli\nDepartment: Computer Science and Engineering\nCreators: Shashank (189CS22045), Dhanush (189CS22013), Vikas (189CS22054)", font=("Arial", 8), fg="white", bg="#1e1e1e")
footer_label.pack(side="bottom", fill="x", pady=10)
footer_label.config(anchor="center")

# Run the GUI
root.mainloop()
