import cv2
import numpy as np

""" Basic operations on images 
    1. Convert the image from the RGB color space to the HSV color space, using an OpenCV color-space converting function.
    2. Modify the image in the HSV color-space by independently scaling (multiplying) the H, S and V channels by 0, 0.2, 0.4, 0.6, and 0.8.
    3. Create a mask image to show all the valid pixel positions where the color difference is less than the threshold of 100.
"""
class Core:
    def __init__(self, img:cv2.Mat) -> None:
        self.img = img


    def core_1(self) -> np.stack:
        print("Running Core 1")

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        rgb_channels = cv2.split(self.img)
        hsv_channels = cv2.split(hsv)

        rgb_combined = np.hstack((rgb_channels[0], rgb_channels[1], rgb_channels[2]))
        hsv_combined = np.hstack((hsv_channels[0], hsv_channels[1], hsv_channels[2])) 

        result = np.vstack((rgb_combined, hsv_combined))

        return result
    

    def core_2(self) -> np.stack:
        print("Running Core 2")

        scales = [0.2, 0.4, 0.6, 0.8]

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        hsv_channels = cv2.split(hsv)

        result = []

        for ind, c in enumerate(hsv_channels):
            for s in scales:
                hsv_mod = hsv.copy()
                hsv_mod[:, :, ind] = cv2.multiply(c,s) # cv2.multiply is used to multiply the all pixel values of the image with the scalar value
                bgr_scaled = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)
                result.append(bgr_scaled)

        h = np.hstack((result[0], result[1], result[2], result[3]))
        s = np.hstack((result[4], result[5], result[6], result[7]))
        v = np.hstack((result[8], result[9], result[10], result[11]))
        final = np.vstack((h, s, v))

        return final
    
    '''
        The color similarity between a pair of pixels can be measured by the distance in
        color space and we can segment an image if we separate pixel pairs based on their distance.
        For the given input image, take the pixel located at index (80, 80) and find all the pixels in the
        image with Euclidean distance of less than 100 in color space
    '''
    def core_3(self):
        print("Running Core 3")
        src = self.img
        ref = src[80,80]

        mask = np.zeros(src.shape[:2], dtype=np.uint8)
    
        height, width = src.shape[:2]
        for y in range(height):
            for x in range(width):
                color = src[y, x]
                dist = cv2.norm(color, ref, cv2.NORM_L2) # sqrt((R1-R2)^2+ (G1-G2)^2+(B1-B2)^2) 

                if dist < 100:
                    mask[y, x] = 255

        result = mask

        return result
    

    def display(self) -> None:
        core_1 = self.core_1()
        core_2 = self.core_2()
        core_3 = self.core_3()

        cv2.imshow("Original", self.img)
        cv2.waitKey(0)
        cv2.imshow("Core 1", core_1)
        cv2.waitKey(0)
        cv2.imshow("Core 2", core_2)
        cv2.waitKey(0)
        cv2.imshow("Core 3", core_3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        