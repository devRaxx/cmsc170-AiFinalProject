import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np

# Load the garbage classification model
processor = AutoImageProcessor.from_pretrained("yangy50/garbage-classification")
model = AutoModelForImageClassification.from_pretrained("yangy50/garbage-classification")

# Load the YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def classify_largest_object(frame):
    # Run YOLOv5 on the frame
    results = yolo_model(frame)
    boxes = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
    
    if len(boxes) == 0:
        return frame, "No objects detected."

    # Calculate the area of each box and find the largest one
    largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))  # (width * height)
    x1, y1, x2, y2, conf, class_idx = largest_box
    
    # Crop the largest object's region from the frame
    largest_object = frame[int(y1):int(y2), int(x1):int(x2)]
    
    # Convert the region to RGB and preprocess for classification
    rgb_frame = cv2.cvtColor(largest_object, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    inputs = processor(images=pil_image, return_tensors="pt")
    
    # Perform classification
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    
    # Determine if the object is biodegradable or non-biodegradable
    if predicted_label in ["cardboard", "paper", "trash"]:
        result = f"Largest Object: {predicted_label} (Biodegradable)"
    else:
        result = f"Largest Object: {predicted_label} (Non-Biodegradable)"
    
    # Draw the bounding box on the frame
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(frame, result, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, result


class GarbageClassifierApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Default video source
        self.video_source = 0

        # Dropdown menu to select camera
        self.camera_label = tk.Label(window, text="Select Camera:")
        self.camera_label.pack(anchor=tk.W)
        self.camera_var = tk.IntVar(value=self.video_source)  # Variable to hold the selected camera index
        self.camera_menu = tk.OptionMenu(window, self.camera_var, *[i for i in range(5)], command=self.change_camera)
        self.camera_menu.pack(anchor=tk.W)

        # Open the video source
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", self.video_source)

        # Create a canvas to display the video frames
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Button to classify the largest object in the current frame
        self.btn_classify = Button(window, text="Classify Largest Object", width=50, command=self.classify_current_frame)
        self.btn_classify.pack(anchor=tk.CENTER, expand=True)

        # Label to display the classification result
        self.label_result = Label(window, text="Classification Result: ")
        self.label_result.pack(anchor=tk.CENTER, expand=True)

        # Start the video loop
        self.delay = 15
        self.update()

        self.window.mainloop()

    def change_camera(self, camera_index):
        # Release the current video source
        if self.vid.isOpened():
            self.vid.release()

        # Set the new video source
        self.video_source = int(camera_index)
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            self.label_result.config(text=f"Unable to open camera {self.video_source}")
        else:
            self.label_result.config(text=f"Camera {self.video_source} selected")

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
            # Classify the largest object (update frame with bounding box)
            processed_frame, _ = classify_largest_object(frame)

            # Convert the frame to RGB format
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.photo = imgtk

        # Repeat after a delay
        self.window.after(self.delay, self.update)

    def classify_current_frame(self):
        # Get the current frame
        ret, frame = self.vid.read()
        if ret:
            # Classify the largest object in the frame
            processed_frame, result = classify_largest_object(frame)
            
            # Convert the frame to RGB format and update the canvas
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.photo = imgtk

            # Display the result
            self.label_result.config(text=result)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


# Create a window and pass it to the Application object
GarbageClassifierApp(tk.Tk(), "Garbage Classifier")
