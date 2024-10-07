from ultralytics import YOLO

# pip install ultralytics==8.2.59
import os
import cv2

# Load the trained YOLOv8 model
model_path = 'standard_errors_objdetection.pt'
model = YOLO(model_path)

# Define the path to the test images
test_images_dir = 'dataset/test/images'

# Create a directory to save the results
results_dir = 'test_results'
os.makedirs(results_dir, exist_ok=True)

# Run inference on each image in the test directory
for image_name in os.listdir(test_images_dir):
    image_path = os.path.join(test_images_dir, image_name)
    image = cv2.imread(image_path)

    # Run the model on the image
    results = model.predict(source=image)

    # Visualize and save the results
    for result in results:
        result_image = result.plot()  # This visualizes the detections on the image
        result_path = os.path.join(results_dir, image_name)
        cv2.imwrite(result_path, result_image)

    print(f"Processed {image_name}, results saved to {result_path}")

print("Testing complete.")
