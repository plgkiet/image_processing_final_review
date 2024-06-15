import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(image_path):
    """ Load an image from a file path. """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    return image

def histogram_equalization(image):
    """ Apply histogram equalization to enhance contrast. """
    return cv2.equalizeHist(image)

def median_filter(image, ksize=5):
    """ Apply median filter to reduce noise. """
    return cv2.medianBlur(image, ksize)

def edge_detection(image):
    """ Apply Canny edge detection to highlight edges. """
    edges = cv2.Canny(image, 100, 200)
    return edges

def plot_comparison(original, processed_images, titles):
    """ Plot the original and processed images for comparison. """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, len(processed_images) + 1, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    for i, (image, title) in enumerate(zip(processed_images, titles), start=2):
        plt.subplot(1, len(processed_images) + 1, i)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.show()

def main(image_path):
    original_image = load_image(image_path)
    
    # Step 1: Histogram Equalization
    hist_eq_image = histogram_equalization(original_image)
    
    # Step 2: Median Filter
    median_filtered_image = median_filter(hist_eq_image)
    
    # Step 3: Edge Detection
    edges_image = edge_detection(median_filtered_image)
    
    # Plot the comparison
    processed_images = [hist_eq_image, median_filtered_image, edges_image]
    titles = ["Histogram Equalization", "Median Filter", "Edge Detection"]
    plot_comparison(original_image, processed_images, titles)

if __name__ == "__main__":
    image_path = '1.jpg'  # Replace with your image path
    main(image_path)
