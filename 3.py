import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    return image

def dilation(image, kernel_size=5):
    """Apply dilation to the image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def erosion(image, kernel_size=5):
    """Apply erosion to the image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def opening(image, kernel_size=5):
    """Apply opening (erosion followed by dilation) to the image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel_size=5):
    """Apply closing (dilation followed by erosion) to the image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def hit_or_miss(image):
    """Apply Hit-or-Miss transform to detect specific patterns."""
    # Define the structuring elements for hit-or-miss
    kernel1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.int32)
    kernel2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32)
    
    hit_or_miss_image = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel1)
    return hit_or_miss_image

def plot_comparison(original, processed_images, titles):
    """Plot the original and processed images for comparison."""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    for i, (image, title) in enumerate(zip(processed_images, titles), start=2):
        plt.subplot(2, 3, i)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.show()

def main(image_path):
    original_image = load_image(image_path)
    
    # Step 1: Dilation
    dilation_image = dilation(original_image)
    
    # Step 2: Erosion
    erosion_image = erosion(original_image)
    
    # Step 3: Opening
    opening_image = opening(original_image)
    
    # Step 4: Closing
    closing_image = closing(original_image)
    
    # Step 5: Hit-or-Miss
    hit_or_miss_image = hit_or_miss(original_image)
    
    # Plot the comparison
    processed_images = [dilation_image, erosion_image, opening_image, closing_image, hit_or_miss_image]
    titles = ["Dilation", "Erosion", "Opening", "Closing", "Hit-or-Miss"]
    plot_comparison(original_image, processed_images, titles)

if __name__ == "__main__":
    image_path = '1.jpg'  # Replace with your image path
    main(image_path)
