import cv2
import numpy as np


""" Histogram Equalization
    Given a gray (single channel) image as input, perform histogram equalization and output the
    result. 
"""
class Challenge:
    def __init__(self, img:cv2.Mat) -> None:
        self.img = img

    
    def histogramEqualization(self) -> cv2.Mat:
        src = self.img # Source image
        hist = cv2.calcHist([src], [0], None, [256], [0, 256]) # Calculate histogram || params: imgs:[src], channels:[0], mask:None, histSize:256, ranges:[0,256]
        cdf = hist.cumsum() # Cumulative Distribution Function
        cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min()) # Normalize
        equalized_image = cdf_normalized[src] # Map intensity values using the normalized CDF
        equalized_image = equalized_image.astype(np.uint8) # Convert to uint8
        return equalized_image


    def display(self) -> None:
        equalized_image = self.histogramEqualization()
        result = np.hstack((self.img, equalized_image))
        cv2.imshow("Challenge", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()