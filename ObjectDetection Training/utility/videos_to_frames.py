import os
import cv2
import random

def extract_frames_from_videos(source_folder, dest_folder, frames_to_extract):

    # Create destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # List all video files in the source folder
    video_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    num_videos = len(video_files)
    if num_videos == 0:
        print("No videos found in the source folder.")
        return
    
    frames_per_video = max(1, frames_to_extract // num_videos)

    for video_file in video_files:

        video_path = os.path.join(source_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        selected_frames = sorted(random.sample(range(total_frames), min(frames_per_video, total_frames)))
        
        for frame_index in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frame_filename = os.path.join(dest_folder, f"{video_file.split('.')[0]}_frame{frame_index}.jpg")
                cv2.imwrite(frame_filename, frame)
        
        cap.release()
    
    print(f"Extracted frames from {num_videos} videos into {dest_folder}")

# Run the function
# extract_frames_from_videos("pipetting", "images", 1000)
extract_frames_from_videos("standard", "images", 1000)