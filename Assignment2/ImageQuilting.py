import numpy as np
import cv2


class ImageQuilting:
    def __init__(
        self,
        image_path: str,
        patch_size: int = 100,
        overlap: int = 20,
        targ_width: int = 500,
    ):
        self.image = cv2.imread(image_path)
        self.patch_size = patch_size
        self.overlap = overlap
        self.target_width = targ_width

    """
    Randomly select a patch from the image with size 100x100
    """
    def random_patch(self)-> np.ndarray:
        max_x = self.image.shape[0] - self.patch_size
        max_y = self.image.shape[1] - self.patch_size
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        return self.image[x:x+self.patch_size, y:y+self.patch_size]  # patch in image from x to x+100, y to y+100 
    
    """
    Find the patch with the minimum sum of squared differences
    """
    def find_min_ssd_patch(self, left_patch: np.ndarray) -> np.ndarray:

        min_ssd = np.inf
        best_patch = None
        rows, cols = self.image.shape[:2]
        # window_shape = (self.patch_size, self.patch_size, self.image.shape[2])
        # stride = self.image.strides  
        # number of bytes to jump-over in the memory to get from one item to the next item along each direction/dimension of the array.
        
        for i in range(0, rows-self.patch_size): # vertical
            for j in range(0, cols - self.patch_size): # horizontal
                candidate = self.image[i:i+self.patch_size, j:j+self.patch_size] # candidate patch (100x100)
                ssd = np.sum((candidate[:, :self.overlap] - left_patch[:, -self.overlap:])**2) # ssd of candidate
                if ssd < min_ssd: # if ssd of candidate is less than min_ssd
                    min_ssd = ssd
                    best_patch = candidate
        return best_patch
    
    """
    Calculate the best seam between two patches
    """
    def calc_best_seam(self, patch1: np.ndarray, patch2: np.ndarray) -> np.ndarray:
        
        # ssd for the overlap regions
        cost = np.sum((patch1[:, -self.overlap:] - patch2[:, :self.overlap])**2, axis=2)
        # Matrix to store cumulative costs
        cumulative_cost = np.zeros_like(cost)
        cumulative_cost[0] = cost[0]

        # Dynamic programming to find the minimum cost path
        for i in range(1, self.patch_size):
            for j in range(self.overlap):
                min_cost = cumulative_cost[i-1, j]
                if j > 0:
                    min_cost = min(min_cost, cumulative_cost[i-1, j-1])
                if j < self.overlap - 1:
                    min_cost = min(min_cost, cumulative_cost[i-1, j+1])
                cumulative_cost[i, j] = cost[i, j] + min_cost

        # Trace back to find the path of minimum cost
        seam = np.zeros(self.patch_size, dtype=np.int32)
        seam[-1] = np.argmin(cumulative_cost[-1])

        for i in range(self.patch_size-2, -1, -1):
            j = seam[i+1]
            if j > 0 and cumulative_cost[i, j-1] < cumulative_cost[i, j]:
                j -= 1
            elif j < self.overlap - 1 and cumulative_cost[i, j+1] < cumulative_cost[i, j]:
                j += 1
            seam[i] = j

        return seam
    
    def blend_seam(self, patch1: np.ndarray, patch2: np.ndarray, seam: np.ndarray) -> np.ndarray:
        blended = np.copy(patch1) 
        for i in range(self.patch_size):  # each row
            blended[i, seam[i]:seam[i]+self.overlap] = patch2[i, seam[i]:seam[i]+self.overlap]

        return blended
    
    def synthesize_texture(self) -> np.ndarray:
        texture = self.random_patch()
        curr_patch = texture
        i = 1

        while texture.shape[1] < self.target_width:
            print(f"Iteration: {i}")
            nxt_patch = self.find_min_ssd_patch(curr_patch)
            seam = self.calc_best_seam(curr_patch, nxt_patch)
            blended_area = self.blend_seam(curr_patch, nxt_patch, seam)
            texture = np.hstack((texture[:, :-self.overlap], blended_area))
            curr_patch = nxt_patch
            i += 1
        
        return texture[:self.patch_size, :self.target_width]
    

if __name__ == "__main__":
    quilter = ImageQuilting("assets/TextureSample.jpg")
    res = quilter.synthesize_texture()

    ex = cv2.imread("assets/QuiltingExample.png")
    cv2.imshow("Texture Synthesis", res)
    cv2.imshow("Original Texture", ex)
    cv2.waitKey(0)