import cv2
import numpy as np

""" Edge Extraction
    Generate three images: 
        * edge magnitude image by Laplacian filter,
        * two images which are the edge filter responses in the x and y dirs (Sobel X, Sobel Y)
"""
class Completion:
    def __init__(self, img:cv2.Mat) -> None:
        self.img = img

        '''Kernels'''
        self.laplacian = np.array([[0, 1, 0], 
                                   [1, -4, 1], 
                                   [0, 1, 0]])
        
        self.sobel_x = np.array([[-1, 0, 1], 
                                 [-2, 0, 2], 
                                 [-1, 0, 1]])
        
        self.sobel_y = np.array([[1, 2, 1], 
                                 [0, 0, 0], 
                                 [-1, -2, -1]])

    def applyFilter(self, kernel:np.array) -> cv2.Mat:
        src = self.img
        
        output = np.zeros_like(src, dtype=np.float32) # init ouput w/ zeros
        
        # Borders
        pad_height = kernel.shape[0] // 2
        pad_width = kernel.shape[1] // 2 

        padded_img = cv2.copyMakeBorder(src, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_REFLECT) #img w/ borders

        for y in range(src.shape[0]):
            for x in range(src.shape[1]):
                # end ind = curr ind + kernel size
                y_end_ind = y + kernel.shape[0] 
                x_end_ind = x + kernel.shape[1]
                region = padded_img[y:y_end_ind, x:x_end_ind] # get region in the padded img || [x to x_end_ind, y to y_end_ind]
                output[y, x] = np.sum(region * kernel) 

        return output

    def display(self):
        #Apply Filters
        laplacian = self.applyFilter(self.laplacian)
        sobel_x = self.applyFilter(self.sobel_x)
        sobel_y = self.applyFilter(self.sobel_y)

        # Normalize
        laplacian = cv2.normalize(laplacian, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        sobel_x = cv2.normalize(sobel_x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        sobel_y = cv2.normalize(sobel_y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        all_results = np.hstack((laplacian, sobel_x, sobel_y))

        print("\n\tLaplacian\t\tSobel X\t\t\tSobel Y")
        cv2.imshow("All Results", all_results)
        cv2.waitKey(0)
        cv2.destroyAllWindows()