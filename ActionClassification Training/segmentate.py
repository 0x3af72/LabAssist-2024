from moviepy.editor import VideoFileClip
import os

def segmentate(video, folder, size):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    clip = VideoFileClip(video)
    duration = clip.duration
    output = []
    for i in range(0, int(duration), size):
        end_time = min(i + size, duration)
        segment = clip.subclip(i, end_time)
        segment.write_videofile(f"{folder}/{i}.mp4", codec="libx264")
        output.append(f"{folder}/{i}.mp4")
    clip.close()
    return output

if __name__ == "__main__":
    segmentate("test.mp4", "output", 2)