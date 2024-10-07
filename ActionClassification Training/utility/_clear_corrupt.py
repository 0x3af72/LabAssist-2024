import os
import subprocess

# Function to check if a file is corrupt using ffmpeg
def is_corrupt(filepath):
    try:
        subprocess.check_output(['ffmpeg', '-v', 'error', '-i', filepath, '-f', 'null', '-'], stderr=subprocess.STDOUT)
        return False  # File is not corrupt
    except subprocess.CalledProcessError:
        return True   # File is corrupt

# Function to delete corrupt MP4 files
def delete_corrupt_mp4(root_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                filepath = os.path.join(foldername, filename)
                if is_corrupt(filepath):
                    print(f"Deleting corrupt file: {filepath}")
                    os.remove(filepath)

# Replace 'root_folder' with the path to the top-level folder you want to start from
delete_corrupt_mp4("/datasets/buretting")
delete_corrupt_mp4("/datasets/pipetting")
delete_corrupt_mp4("/datasets/swirling")
delete_corrupt_mp4("/datasets/standard")