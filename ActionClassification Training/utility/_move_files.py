import os
import random
import shutil

def move_files(original, destination, num):
    # Ensure the original and destination directories exist
    if not os.path.exists(original):
        print(f"Original directory '{original}' does not exist.")
        return
    if not os.path.exists(destination):
        os.makedirs(destination)

    # Get a list of all files in the original directory
    files = [f for f in os.listdir(original) if os.path.isfile(os.path.join(original, f))]

    # Check if the number of files to move is greater than the available files
    if num > len(files):
        print(f"Requested number of files to move ({num}) exceeds available files ({len(files)}).")
        num = len(files)

    # Randomly select files to move
    files_to_move = random.sample(files, num)

    # Move each file to the destination directory
    for file_name in files_to_move:
        src_path = os.path.join(original, file_name)
        dst_path = os.path.join(destination, file_name)
        shutil.move(src_path, dst_path)
        print(f"Moved: {file_name}")
        
def count_files(folder):
    print(folder, len(os.listdir(folder)))

# count_files("datasets/vf_shaking_wrong")
move_files("datasets/vf_shaking_wrong_undersample", "datasets/vf_shaking_wrong", 300)