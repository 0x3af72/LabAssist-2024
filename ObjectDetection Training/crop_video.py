from ultralytics import YOLO
from moviepy.editor import *
from PIL import Image
import threading
import os
import cv2
import time

# pip install ultralytics==8.2.59 moviepy

model_path = "standard_errors_objdetection.pt"
model = YOLO(model_path)

def even(n):
    n = int(n)
    if n % 2 != 0:
        return n - 1
    return n

def scale_box(x1, y1, x2, y2, scale_factor, min_x1, min_y1, max_x2, max_y2):
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    width = x2 - x1
    height = y2 - y1

    new_width = width * scale_factor
    new_height = height * scale_factor

    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    # Adjust to ensure the box stays within the bounds
    new_x1 = max(new_x1, min_x1)
    new_y1 = max(new_y1, min_y1)
    new_x2 = min(new_x2, max_x2)
    new_y2 = min(new_y2, max_y2)

    return new_x1, new_y1, new_x2, new_y2

j = 0
def crop_video(file):
    global j
    
    video = VideoFileClip(file)
    second = 0
    while second < video.duration:
        
        start = time.time()
        
        clip = video.subclip(second, min(second + 2, video.duration))
        frame = cv2.cvtColor(clip.get_frame(0), cv2.COLOR_RGB2BGR)
        prediction = predict_image(frame)
        del frame
        
        # handle no predictions
        if prediction is None:
            print("NONE:", j)
            second += 2
            j += 1
            continue
        
        # crop out
        x1, y1, x2, y2 = scale_box(*prediction[3], 1.1, 0, 0, *clip.size)
        clip = clip.crop(
            x1=even(x1), y1=even(y1),
            x2=even(x2), y2=even(y2)
        )
        print(f"FOR {j}")
        print(x1, y1, x2, y2)
            
        clip.write_videofile(f"test_results/ok_{j}.mp4", codec="libx264", audio=False)
        clip.close()
        j += 1
        second += 2
        
        print("TIME TAKEN:", time.time() - start)
        
def convert(file):
    
    outfile = "a" + file
    
    clip = VideoFileClip(file)
    second = 0
        
    start = time.time()
    
    frame = cv2.cvtColor(clip.get_frame(0), cv2.COLOR_RGB2BGR)
    prediction = predict_image(frame)
    del frame

    # handle no predictions
    if prediction is None:
        print("NONE:", j)
        second += 2
        j += 1
        return

    # crop out
    x1, y1, x2, y2 = scale_box(*prediction[3], 1.1, 0, 0, *clip.size)
    clip = clip.crop(
        x1=even(x1), y1=even(y1),
        x2=even(x2), y2=even(y2)
    )
    
    # scale
    width = x2 - x1
    height = y2 - y1
    if width > height:
        height = height / width * 640
        width = 640
    else:
        width = width / height * 640
        height = 640
    clip = clip.resize(width=width, height=height)
    clip = CompositeVideoClip([ImageClip("black.png", duration=clip.duration), clip.set_position("center")])
    
    # save it
    clip.write_videofile(outfile, codec="libx264", audio=False)
    clip.close()

    print("TIME TAKEN:", time.time() - start)

i = 0
def predict_image(image):
    global i

    # model prediction
    results = model.predict(source=image)
    
    # parse results
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    boxes = results[0].boxes.xyxy.tolist()
    predictions = list(zip(classes, names, confidences, boxes))
    
    for result in results:
        result_image = result.plot()
        cv2.imwrite(f"test_results/{i}.jpg", result_image)
        i += 1
        break
    
    if not predictions:
        return None
    prediction = max(predictions, key=lambda x: x[2]) # highest confidence
    
    return prediction

if __name__ == "__main__":
    
    # crop_video("tapping_demo.mp4")
    for f in os.listdir("vf_shaking_correct"):
        try:
            convert(f"vf_shaking_correct/{f}")
        except Exception as e:
            print(e)
            print(f"vf_shaking_correct/{f}")
    # for f in os.listdir("videos/vf_shaking_correct"):
    #     try:
    #         convert(f"videos/vf_shaking_correct/{f}")
    #     except Exception as e:
    #         print(e)
    #         print(f"videos/vf_shaking_correct/{f}")
    # for f in os.listdir("videos/vf_shaking_wrong"):
    #     try:
    #         convert(f"videos/vf_shaking_wrong/{f}")
    #     except Exception as e:
    #         print(e)
    #         print(f"videos/vf_shaking_wrong/{f}")