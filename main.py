import cv2
import numpy as np
import os
import time
from tqdm import tqdm

def compute_disparity(img1, img2):
    """Compute the disparity map using OpenCV's StereoBM or StereoSGBM."""
    # Create StereoBM or StereoSGBM object for disparity calculation
    stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
    disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0
    
    # Normalize the disparity map for better visualization and saving
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return disparity_normalized

def disparity_to_depth(disparity_map, baseline, focal_length):
    """Convert disparity map to depth map using baseline and focal length."""
    depth_map = np.zeros(disparity_map.shape, np.float32)
    
    # Avoid small disparity values causing unrealistic depth
    disparity_map[disparity_map < 1] = 1  # Set small disparity values to 1
    
    # Depth calculation: depth = baseline * focal_length / disparity
    depth_map = (baseline * focal_length) / disparity_map
    
    # Normalize the depth map for visualization
    depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return depth_map

def save_results(disparity_map, depth_map, output_folder, dataset_number):
    """Save disparity and depth maps to the specified dataset folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    disparity_file = os.path.join(output_folder, f"disparity_map_{dataset_number}.png")
    depth_file = os.path.join(output_folder, f"depth_map_{dataset_number}.png")

    cv2.imwrite(disparity_file, disparity_map)
    cv2.imwrite(depth_file, depth_map)

    print(f"Disparity map saved: {disparity_file}")
    print(f"Depth map saved: {depth_file}")

def main():
    dataset_number = int(input("Please enter the dataset number (1/2/3) to use for calculating the depth map\n"))
    
    # Paths for the left and right images in the dataset folder
    img1_path = f"Dataset{dataset_number}/im0.png"
    img2_path = f"Dataset{dataset_number}/im1.png"
    
    # Read images in grayscale mode
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Error: Images not found in Dataset{dataset_number}.")
        return
    
    # Resize images to speed up computation
    width = int(img1.shape[1] * 0.3)
    height = int(img1.shape[0] * 0.3)
    img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)

    # Define camera parameters (baseline and focal length for each dataset)
    baselines = [177.288, 144.049, 174.019]
    focal_lengths = [5299.313, 4396.869, 5806.559]

    baseline = baselines[dataset_number - 1]
    focal_length = focal_lengths[dataset_number - 1]

    # Start time measurement
    start_time = time.time()

    print("Computing disparity map...")
    
    # Adding progress bar for the disparity computation
    with tqdm(total=100, desc="Disparity Calculation", ncols=100) as pbar:
        disparity_map = compute_disparity(img1, img2)
        pbar.update(100)  # Update progress bar to 100% when done

    print("Converting disparity to depth map...")
    depth_map = disparity_to_depth(disparity_map, baseline, focal_length)

    # Save the results
    save_results(disparity_map, depth_map, f"Dataset{dataset_number}/", dataset_number)

    # End time measurement
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

