import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(image_path):
    """Load an input image from a file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    return image

def edge_detection(image):
    """Apply Canny edge detection to find edges."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return edges

def corner_detection(image):
    """Use the Harris corner detector to identify corners."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)
    corners = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]
    return image

def blob_detection(image):
    """Implement blob detection to find blob-like structures."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(gray_image)
    blob_image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return blob_image

def plot_comparison(original, processed_images, titles):
    """Plot the original and processed images for comparison."""
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    for i, (image, title) in enumerate(zip(processed_images, titles), start=2):
        plt.subplot(2, 2, i)
        plt.imshow(image, cmap='gray' if title == "Edges" else None)
        plt.title(title)
        plt.axis('off')
    
    plt.show()

def main(image_path):
    original_image = load_image(image_path)
    
    # Step 1: Edge detection
    edges = edge_detection(original_image)
    
    # Step 2: Corner detection
    corners_image = corner_detection(original_image.copy())
    
    # Step 3: Blob detection
    blobs_image = blob_detection(original_image.copy())
    
    # Plot the comparison
    processed_images = [edges, corners_image, blobs_image]
    titles = ["Edges", "Corners", "Blobs"]
    plot_comparison(original_image, processed_images, titles)

if __name__ == "__main__":
    image_path = '1.jpg'  # Replace with your image path
    main(image_path)
