import cv2
import os
import time

# --- Configuration ---
INPUT_DIR = '/content/data_set'
OUTPUT_DIR = 'output_seq'
IMG_SIZE = (128, 128)
WATERMARK_TEXT = 'Processed'
WATERMARK_COLOR = (255, 255, 255)  # White
WATERMARK_FONT = cv2.FONT_HERSHEY_SIMPLEX
WATERMARK_SCALE = 0.5
WATERMARK_THICKNESS = 1

# --- Main Script ---

# Record the start time
start_time = time.time()

# Create the main output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Loop through each class folder in the input directory
for class_folder in os.listdir(INPUT_DIR):
    class_folder_path = os.path.join(INPUT_DIR, class_folder)

    # Create a corresponding class folder in the output directory
    output_class_folder = os.path.join(OUTPUT_DIR, class_folder)
    if not os.path.exists(output_class_folder):
        os.makedirs(output_class_folder)

    # Loop through each image in the class folder
    for image_name in os.listdir(class_folder_path):
        image_path = os.path.join(class_folder_path, image_name)

        # 1. Read the image
        image = cv2.imread(image_path)
        if image is not None:
            # 2. Resize the image
            resized_image = cv2.resize(image, IMG_SIZE)

            # 3. Add a watermark
            # Get the position for the watermark (bottom-left corner)
            text_size, _ = cv2.getTextSize(WATERMARK_TEXT, WATERMARK_FONT, WATERMARK_SCALE, WATERMARK_THICKNESS)
            text_x = 10
            text_y = IMG_SIZE[1] - 10
            cv2.putText(resized_image, WATERMARK_TEXT, (text_x, text_y), WATERMARK_FONT, WATERMARK_SCALE, WATERMARK_COLOR, WATERMARK_THICKNESS)


            # 4. Save the processed image
            output_image_path = os.path.join(output_class_folder, image_name)
            cv2.imwrite(output_image_path, resized_image)

# Record the end time
end_time = time.time()

# Calculate and print the total execution time
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")
