import cv2
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
INPUT_DIR = '/content/data_set'
OUTPUT_DIR = 'output_parallel'
IMG_SIZE = (128, 128)
WATERMARK_TEXT = 'Processed'
WATERMARK_COLOR = (255, 255, 255)  # White
WATERMARK_FONT = cv2.FONT_HERSHEY_SIMPLEX
WATERMARK_SCALE = 0.5
WATERMARK_THICKNESS = 1

# A list of worker counts to test
WORKER_COUNTS = [1, 2, 4, 8]

# --- Helper Function: Get All Image Paths ---
def get_all_image_paths(input_dir, output_dir):
    """
    Finds all images, creates corresponding output directories,
    and returns a list of (input_path, output_path) tuples.
    """
    image_paths_list = []
    
    # Create the main output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each class folder
    for class_folder in os.listdir(input_dir):
        class_folder_path = os.path.join(input_dir, class_folder)
        
        if os.path.isdir(class_folder_path):
            # Create a corresponding class folder in the output directory
            output_class_folder = os.path.join(output_dir, class_folder)
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)
                
            # Loop through each image
            for image_name in os.listdir(class_folder_path):
                input_path = os.path.join(class_folder_path, image_name)
                output_path = os.path.join(output_class_folder, image_name)
                
                # Add the pair to our list of work
                image_paths_list.append((input_path, output_path))
                
    return image_paths_list

# --- Worker Function: Process a Single Image ---
def process_image(paths):
    """
    Reads, resizes, watermarks, and saves a single image.
    This is the "unit of work" for each thread.
    """
    input_path, output_path = paths
    
    try:
        # 1. Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Warning: Could not read {input_path}, skipping.")
            return

        # 2. Resize the image
        resized_image = cv2.resize(image, IMG_SIZE)

        # 3. Add a watermark
        (text_width, text_height), _ = cv2.getTextSize(WATERMARK_TEXT, WATERMARK_FONT, WATERMARK_SCALE, WATERMARK_THICKNESS)
        text_x = 10
        text_y = IMG_SIZE[1] - 10
        cv2.putText(resized_image, WATERMARK_TEXT, (text_x, text_y), WATERMARK_FONT, WATERMARK_SCALE, WATERMARK_COLOR, WATERMARK_THICKNESS)

        # 4. Save the processed image
        cv2.imwrite(output_path, resized_image)
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# --- Main Script ---
def main():
    # First, get the full list of all images to be processed
    print(f"Discovering all images in {INPUT_DIR}...")
    all_image_paths = get_all_image_paths(INPUT_DIR, OUTPUT_DIR)
    print(f"Found {len(all_image_paths)} images to process.")
    
    results = []

    # Now, run the processing with different worker counts
    for num_workers in WORKER_COUNTS:
        print(f"\n--- Testing with {num_workers} worker(s) ---")
        start_time = time.time()
        
        # Use ThreadPoolExecutor to parallelize the work
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 'map' applies the 'process_image' function to every item 
            # in the 'all_image_paths' list, distributing the work
            # among the threads in the pool.
            list(executor.map(process_image, all_image_paths))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Time taken: {total_time:.2f} seconds")
        results.append({'workers': num_workers, 'time': total_time})

    # --- Display Speedup Table ---
    print("\n--- Speedup Table ---")
    print("Workers | Time (s) | Speedup")
    print("-------- | -------- | -------")
    
    # Get the baseline time (time taken by 1 worker)
    baseline_time = results[0]['time']
    
    for res in results:
        workers = res['workers']
        time_taken = res['time']
        speedup = baseline_time / time_taken
        
        # Format the table
        print(f"{workers:<8} | {time_taken:<8.2f} | {speedup:.2f}x")

# Run the main function
if __name__ == "__main__":
    main()