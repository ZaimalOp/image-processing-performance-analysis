import cv2
import os
import time
import numpy as np
import multiprocessing

# --- Configuration ---
INPUT_DIR = '/content/data_set'
OUTPUT_DIR = 'output_distributed'
IMG_SIZE = (128, 128)
WATERMARK_TEXT = 'Processed'
WATERMARK_COLOR = (255, 255, 255)
WATERMARK_FONT = cv2.FONT_HERSHEY_SIMPLEX
WATERMARK_SCALE = 0.5
WATERMARK_THICKNESS = 1
NUM_NODES = 2 # Simulating 2 machines

# --- Reusable Function to Process a Single Image ---
def process_single_image(input_path, output_path):
    """Core logic to process one image."""
    try:
        image = cv2.imread(input_path)
        if image is not None:
            resized_image = cv2.resize(image, IMG_SIZE)
            cv2.putText(resized_image, WATERMARK_TEXT, (10, IMG_SIZE[1] - 10), 
                        WATERMARK_FONT, WATERMARK_SCALE, WATERMARK_COLOR, WATERMARK_THICKNESS)
            cv2.imwrite(output_path, resized_image)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# --- Node Function (Worker) ---
def node_task(node_id, image_subset, results_dict):
    """
    This function simulates a single node.
    It processes its assigned images and records its own execution time.
    """
    print(f"Node {node_id}: Starting processing...")
    start_time = time.time()
    
    for input_path, output_path in image_subset:
        process_single_image(input_path, output_path)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    # Report results back to the master using the shared dictionary
    results_dict[node_id] = {'count': len(image_subset), 'time': total_time}
    print(f"Node {node_id}: Finished in {total_time:.2f}s")


# --- Main Script (Master) ---
def main():
    # 1. Get a list of all image paths and create output directories
    all_image_paths = []
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for class_folder in os.listdir(INPUT_DIR):
        class_folder_path = os.path.join(INPUT_DIR, class_folder)
        if os.path.isdir(class_folder_path):
            output_class_folder = os.path.join(OUTPUT_DIR, class_folder)
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)
            for image_name in os.listdir(class_folder_path):
                input_path = os.path.join(class_folder_path, image_name)
                output_path = os.path.join(output_class_folder, image_name)
                all_image_paths.append((input_path, output_path))
    
    print(f"Total images found: {len(all_image_paths)}")

    # --- BASELINE: Run sequentially to calculate speedup ---
    print("\nRunning sequential processing for baseline...")
    seq_start_time = time.time()
    for input_path, output_path in all_image_paths:
        process_single_image(input_path, output_path)
    seq_end_time = time.time()
    sequential_time = seq_end_time - seq_start_time
    print(f"Sequential time: {sequential_time:.2f} seconds")
    
    # --- SIMULATION: Run distributed processing ---
    print(f"\nStarting distributed simulation with {NUM_NODES} nodes...")
    
    # 2. Divide the work (list of images) among the nodes
    image_subsets = np.array_split(all_image_paths, NUM_NODES)

    # 3. Set up the Manager for inter-process communication
    with multiprocessing.Manager() as manager:
        # This dictionary is shared between all processes
        results_dict = manager.dict()
        processes = []
        
        # Master timer starts now
        master_start_time = time.time()
        
        # 4. Create and start a process for each node
        for i in range(NUM_NODES):
            node_id = i + 1
            subset = image_subsets[i]
            process = multiprocessing.Process(target=node_task, args=(node_id, subset, results_dict))
            processes.append(process)
            process.start()
            
        # 5. Wait for all nodes to finish their work
        for process in processes:
            process.join() # This blocks until the process is done
        
        # Master timer ends after the last node has finished
        master_end_time = time.time()
        total_distributed_time = master_end_time - master_start_time
        
        # 6. Print the summary
        print("\n--- Distributed Processing Summary ---")
        for node_id, result in sorted(results_dict.items()):
            print(f"Node {node_id} processed {result['count']} images in {result['time']:.1f}s")
        
        print(f"Total distributed time: {total_distributed_time:.1f}s")
        
        efficiency = sequential_time / total_distributed_time
        print(f"Efficiency: {efficiency:.2f}x over sequential")

# The __name__ == '__main__' guard is crucial for multiprocessing
if __name__ == '__main__':
    main()