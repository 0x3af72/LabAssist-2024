import os
from PIL import Image

def resize_images_in_folder(folder_path, size=(640, 640)):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Get all files in the folder
    for filename in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        try:
            with Image.open(file_path) as img:
                # Resize the image
                img_resized = img.resize(size, Image.ANTIALIAS)
                
                # Save the resized image, overwriting the original
                img_resized.save(file_path)
                
                print(f"Resized {filename}")
        except IOError:
            # If the file is not an image, skip it
            print(f"Skipping {filename}, not an image file.")

# Usage
folder_path = "images"  # Replace with your folder path
resize_images_in_folder(folder_path)
