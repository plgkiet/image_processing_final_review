import cv2
import numpy as np

def frame_differencing(prev_frame, cur_frame):
    """Compute the absolute difference between two frames."""
    diff = cv2.absdiff(prev_frame, cur_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    return thresh

def background_subtraction(bg_subtractor, frame):
    """Apply background subtraction to isolate moving objects."""
    fg_mask = bg_subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    return fg_mask

def morphological_operations(mask):
    """Use morphological operations to clean up the segmented output."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    return cleaned_mask

def main(video_path=None):
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Use the camera

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame differencing
        diff_mask = frame_differencing(prev_frame, frame)
        
        # Background subtraction
        fg_mask = background_subtraction(bg_subtractor, frame)
        
        # Morphological operations
        cleaned_diff_mask = morphological_operations(diff_mask)
        cleaned_fg_mask = morphological_operations(fg_mask)
        
        # Combine masks for visualization
        combined_mask = cv2.bitwise_or(cleaned_diff_mask, cleaned_fg_mask)
        
        # Display results
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Frame Differencing', cleaned_diff_mask)
        cv2.imshow('Background Subtraction', cleaned_fg_mask)
        cv2.imshow('Combined Mask', combined_mask)
        
        prev_frame = frame
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = None  # Replace with your video path or None for webcam
    main(video_path)
