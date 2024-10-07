from model import *
import os

import torch
import cv2
import numpy as np
from model import VideoClassifier
from torchvision import transforms

from segmentate import *

# pip install moviepy

MODEL_FILE = "standard_final.pth"
# MODEL_FILE = "splitter_final.pth"

model = VideoClassifier()
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes_splitter = ["buretting", "pipetting", "swirling", "standard"]
classes_standard = ["rotating_vf", "tapping_vf", "vf_shaking_correct", "vf_shaking_wrong"]
classes = classes_splitter if MODEL_FILE == "splitter_final.pth" else classes_standard 

def predict_frame(file):

    cap = cv2.VideoCapture(file)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)
    
    cap.release()
    # Shape should be (num_frames, 3, 640, 640)
    video_tensor = torch.stack(frames)  # shape: (num_frames, 3, 640, 640)
    
    # Add batch dimension and time dimension
    video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # shape: (1, 3, num_frames, 640, 640)

    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(video_tensor)

    print("outputs:", outputs)
    
    # Get the predicted class
    predicted_class = torch.argmax(outputs, dim=1).item()
    
    del video_tensor, frames, frame
    torch.cuda.empty_cache()
    
    return predicted_class, outputs

def predict_video(file):
    frames = segmentate(file, f"{os.path.basename(file)}.segments", 2)
    # frames = [f"standard_correct.mp4.segments/{x}" for x in os.listdir("standard_correct.mp4.segments")]
    seconds = 0
    for frame in frames:
        predicted_class, outputs = predict_frame(frame)
        print(f"{seconds}s: {classes[predicted_class]} ({outputs})")
        seconds += 2
        
def test(csv):
    
    with open(csv, "r") as r:
        data = r.read()
    data = data.strip().split("\n")
    
    types = {}
    for line in data:
        
        video_path, label = line.split(" ")
        label = int(label)
        
        if label in types:
            types[label]["count"] += 1
        else:
            types[label] = {"count": 1, "correct": 0}
            
        prediction, outputs = predict_frame(video_path)
        if label != prediction:
            print(f"FRAME: {video_path}, CORRECT: {label}, PREDICTION: {prediction}")
        else:
            print(f"CORRECT({label})")
        
        if label == prediction:
            types[label]["correct"] += 1
            
        # Clean up
        del prediction, outputs
        torch.cuda.empty_cache()
            
    for label in types:
        print(f"============ {label} ============")
        print(f"Total: {types[label]['count']}")
        print(f"Correct: {types[label]['correct']}")
        print(f"Wrong: {types[label]['count'] - types[label]['correct']}")
        print(f"Accuracy: {types[label]['correct'] / types[label]['count'] * 100}%")

# print(predict_video("standard_correct.mp4"))
test("ok.csv")
test("ok2.csv")
test("test.csv")