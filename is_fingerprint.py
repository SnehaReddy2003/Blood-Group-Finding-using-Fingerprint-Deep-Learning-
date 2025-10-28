import cv2
import numpy as np

def is_fingerprint(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    
    # Edge metrics
    edge_count = np.sum(edges > 0)
    total_pixels = edges.size
    edge_percentage = (edge_count / total_pixels) * 100
    
    # Stricter thresholds
    min_edge_percent = 5.0
    max_edge_percent = 20.0
    threshold = total_pixels * (min_edge_percent / 100)
    
    # Variance and ridge frequency
    variance = np.var(gray)
    roi = gray[gray.shape[0]//4:3*gray.shape[0]//4, gray.shape[1]//4:3*gray.shape[1]//4]
    freq = np.fft.fft2(roi)
    freq_power = np.abs(freq) ** 2
    ridge_freq = np.mean(freq_power) > 1000
    
    # Debugging output
    print(f"Edge count: {edge_count}, Total pixels: {total_pixels}")
    print(f"Edge percentage: {edge_percentage:.2f}%")
    print(f"Image variance: {variance}")
    print(f"Ridge frequency check: {ridge_freq}")
    print(f"Dynamic threshold: {threshold}")
    
    # Fingerprint criteria
    is_fp = (edge_count > threshold and 
             min_edge_percent <= edge_percentage <= max_edge_percent and 
             variance > 100 and 
             ridge_freq)
    
    print(f"Is fingerprint? {is_fp}")
    return is_fp