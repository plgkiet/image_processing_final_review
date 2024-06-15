import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    return image

def otsu_thresholding(image):
    """Apply Otsu's thresholding method to separate foreground and background."""
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def region_based_segmentation(image):
    """Implement region-based segmentation using watershed algorithm."""
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown == 0] = 0
    
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_color, markers)
    image_color[markers == -1] = [255, 0, 0]
    
    return image_color

def edge_detection(image):
    """Apply Canny edge detection to highlight edges."""
    edges = cv2.Canny(image, 100, 200)
    return edges

def plot_comparison(original, processed_images, titles):
    """Plot the original and processed images for comparison."""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    for i, (image, title) in enumerate(zip(processed_images, titles), start=2):
        plt.subplot(2, 2, i)
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis('off')
    
    plt.show()

def main(image_path):
    original_image = load_image(image_path)
    
    # Step 1: Otsu's Thresholding
    otsu_image = otsu_thresholding(original_image)
    
    # Step 2: Region-based Segmentation
    region_segmented_image = region_based_segmentation(original_image)
    
    # Step 3: Edge Detection
    edges_image = edge_detection(original_image)
    
    # Plot the comparison
    processed_images = [otsu_image, region_segmented_image, edges_image]
    titles = ["Otsu Thresholding", "Region-based Segmentation", "Edge Detection"]
    plot_comparison(original_image, processed_images, titles)

if __name__ == "__main__":
    image_path = '1.jpg'  # Replace with your image path
    main(image_path)
