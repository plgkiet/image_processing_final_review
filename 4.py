import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(image_path):
    """Load a color image from a file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    return image

def convert_color_spaces(image):
    """Convert the image to different color spaces (HSV, LAB)."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return hsv_image, lab_image

def apply_pseudocolor(image):
    """Apply pseudocolor transformation to highlight specific features."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pseudocolor = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    return pseudocolor

def enhance_color(image):
    """Enhance the color contrast using histogram equalization."""
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb_image)
    cv2.equalizeHist(channels[0], channels[0])
    merged = cv2.merge(channels)
    enhanced_image = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return enhanced_image

def plot_comparison(original, processed_images, titles):
    """Plot the original and processed images for comparison."""
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    for i, (image, title) in enumerate(zip(processed_images, titles), start=2):
        plt.subplot(2, 3, i)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.show()

def main(image_path):
    original_image = load_image(image_path)
    
    # Step 1: Convert to different color spaces
    hsv_image, lab_image = convert_color_spaces(original_image)
    
    # Step 2: Apply pseudocolor transformation
    pseudocolor_image = apply_pseudocolor(original_image)
    
    # Step 3: Enhance color contrast
    enhanced_image = enhance_color(original_image)
    
    # Plot the comparison
    processed_images = [hsv_image, lab_image, pseudocolor_image, enhanced_image]
    titles = ["HSV Image", "LAB Image", "Pseudocolor Image", "Enhanced Color"]
    plot_comparison(original_image, processed_images, titles)

if __name__ == "__main__":
    image_path = '1.jpg'  # Replace with your image path
    main(image_path)
