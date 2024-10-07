# pip install pytorchvideo pytorch-lightning ultralytics==8.2.59 moviepy

from crop_video import *

from videoclassifier import *
import torch
import cv2
from torchvision import transforms

from moviepy.editor import VideoFileClip
import os
import numpy as np

import random
from collections import Counter
import threading
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main model
main_model = VideoClassifier()
main_model.load_state_dict(torch.load("standard-videoclassification.pth"))
main_model.eval()
main_model.to(device)
main_model_classes = ["tapping_vf", "vf_shaking_correct", "vf_shaking_wrong"]
main_model_classes_diff = {"tapping_vf": 0.1, "vf_shaking_correct": -69, "vf_shaking_wrong": 0}
acceptable = {"hand_on_vf": ["vf_shaking_correct", "vf_shaking_wrong"], "pipette_in_vf": ["tapping_vf"]}

# Sliding window - count occurrences in sublist of length N
def count_max_occurrences(lst, window_length, *elements):
    
    current_count = 0
    for element in elements:
        current_count += lst[:window_length].count(element)
    max_occurrences = current_count
    
    for i in range(1, len(lst) - window_length + 1):
        if lst[i - 1] in elements:
            current_count -= 1
        if lst[i + window_length - 1] in elements:
            current_count += 1
        max_occurrences = max(max_occurrences, current_count)
    
    return max_occurrences

# Find errors given a video
def standard_errors(file): 
    
    frames = []
    frame_predictions = {}
    frame_predictions_diff = {c: [] for c in main_model_classes}
    expect_frames = 0

    # generate clips and create threads
    frame_num = 0
    clips = crop_video(file)
    while True:
        try:
            frame = f"frame_{frame_num}"
            print(frame)
            output = next(clips)
            frame_num += 2
            if not output:
                continue
            clip, objdet_class = output
            expect_frames += 1
            frames.append(frame)
            # clip.write_videofile("ok/"+frame+".mp4")
            predict_frame_thread(frame_predictions, frame_predictions_diff, clip, frame, objdet_class)
            # threading.Thread(target=predict_frame_thread, args=(frame_predictions, frame_predictions_diff, clip, frame)).start()
        except StopIteration:
            break
    
    # Wait for all predictions to be complete
    while expect_frames != len(frame_predictions):
        time.sleep(0.1)
        
    for c in frame_predictions_diff:
        if frame_predictions_diff[c]:
            frame_predictions_diff[c] = sum(frame_predictions_diff[c]) / len(frame_predictions_diff[c])

    for frame in frames:
        prediction = frame_predictions[frame]
        if prediction is not None: 
            diff_avg_over = prediction[1] - frame_predictions_diff[prediction[0]]
        else:
            diff_avg_over = None
        # if diff_avg_over < main_model_classes_diff[prediction[0]]:
        #     frame_predictions[frame] = (None, 0)
        print(f"{frame}: {frame_predictions[frame][0] if prediction else prediction} ({diff_avg_over})")
    
    frame_predictions_values = [x[0] for x in list(frame_predictions.values()) if x is not None]
    
    # VF SHAKING ERROR:
    # - no presence of correct (4 frames of correct in 8)
    # - presence of wrong (3 frames of wrong in 6)
    vf_shaking_error = ()
    correct_frames = count_max_occurrences(frame_predictions_values, 8, "vf_shaking_correct")
    wrong_frames = count_max_occurrences(frame_predictions_values, 6, "vf_shaking_wrong")
    if correct_frames >= 4: # 3 correct frames
        vf_shaking_error = (False, "")
    elif wrong_frames >= 3: # 3 wrong frames
        vf_shaking_error = (True, "Incorrect volumetric flask shaking technique was used. You need to turn the flask upside down and shake it.")
    else: # no correct or wrong, no shaking
        vf_shaking_error = (True, "No volumetric flask shaking was detected.")
        
    # LAST DROP ERROR:
    # - no presence of tapping (2 frames of tapping and rotating in 6)
    last_drop_error = ()
    tapping_frames = count_max_occurrences(frame_predictions_values, 6, "tapping_vf")
    if tapping_frames >= 2: # 2 correct frames
        last_drop_error = (False, "")
    else:
        last_drop_error = (True, "Failure to remove last drop from pipette. No tapping or rotating of the pipette on the volumetric flask was detected.")
    
    return {"vf_shaking_error": vf_shaking_error, "last_drop_error": last_drop_error}

# Threaded wrapper for predict_frame
def predict_frame_thread(frame_predictions, frame_predictions_diff, clip, frame, objdet_class):
    prediction, diff = predict_frame(clip)
    if not prediction in acceptable[objdet_class]:
        frame_predictions[frame] = None
        return
    frame_predictions[frame] = (prediction, diff)
    frame_predictions_diff[prediction].append(diff)
    print(f"{frame}: {frame_predictions[frame]}\n")

# Given a 2s frame, determine if it is "buretting" or "pipetting" or "swirling" or "standard"
def predict_frame(video):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames = []
    for frame in video.iter_frames():
        frame = transform(frame)
        frames.append(frame)
    video_tensor = torch.stack(frames)  # shape: (num_frames, 3, 640, 640)
    video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # shape: (1, 3, num_frames, 640, 640)
    video_tensor = video_tensor.to(device)
    
    start = time.time()
    with torch.no_grad():
        outputs = main_model(video_tensor)
    print(f"Prediction took: {time.time() - start}s")

    prediction_idx = torch.argmax(outputs, dim=1).item()
    prediction = main_model_classes[prediction_idx]
    confidence = outputs[0][prediction_idx].item()
    diff = (outputs[0][prediction_idx] * 2 - sum(outputs[0])).item()

    print("PREDICTION:", prediction)
    print("OUTPUTS:", outputs)
    print("DIFF", (outputs[0][prediction_idx] * 2 - sum(outputs[0])).item())

    del video_tensor, frames
    torch.cuda.empty_cache()
    
    # if confidence < main_model_classes_confidence[prediction_idx]:
    #     prediction = None

    return prediction, diff

if __name__ == "__main__":
    
    print("Hello world")
    # print(standard_errors("videos/all_demo.mp4"))
    '''
    tapping vf: 24s-44s
    wrong shaking: 60s-70s
    correct shaking: 72s-82s
    '''
    print(standard_errors("videos/shaking_demo.mp4"))
    print(standard_errors("videos/tapping_vf_demo.mp4"))
    print(standard_errors("videos/all_correct.mp4")) # tapping/rotating: 1:00 - 1:12 (60, 72), 180s for correct shaking
    # print(standard_errors("shaking_error_2.mp4")) # wrong tech at 238s, lastdrop at 74s
    # print(standard_errors("shaking_error.mp4")) # 02:44, 02:46 (164, 166)
    # print(standard_errors("last_drop_error.mp4")) # correct shaking at 218 to 221
