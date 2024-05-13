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

    def random_patch(self) -> np.ndarray:
        max_x = self.image.shape[0] - self.patch_size
        max_y = self.image.shape[1] - self.patch_size
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        return self.image[
            x : x + self.patch_size, y : y + self.patch_size
        ]  # patch in image from x to x+100, y to y+100

    """
    Find the patch with the minimum sum of squared differences
    """
    def find_min_ssd_patch(self, left_patch: np.ndarray) -> np.ndarray:
        best_patch = None
        min_ssd = np.inf

        for _ in range(50):  # sample 50 random patches
            candidate_patch = self.random_patch()
            overlap_region_candidate = candidate_patch[:, : self.overlap]
            error = np.sum(
                (overlap_region_candidate - left_patch[:, -self.overlap :]) ** 2
            )
            if error < min_ssd:
                min_ssd = error
                best_patch = candidate_patch

        return best_patch

    """
    Calculate the best seam between two patches
    """

    def calc_best_seam(self, curr: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        rows, cols = curr.shape[:2]
        error = (curr - nxt) ** 2
        cost = np.sum(error, axis=2)  # Sum error across color channels

        # Initialize the dynamic programming table
        dp = np.zeros_like(cost)
        dp[0, :] = cost[0, :]

        # Populate the dp table
        for i in range(1, rows):
            for j in range(cols):
                min_cost = dp[i - 1, j]  # Directly above
                if j > 0:
                    min_cost = min(min_cost, dp[i - 1, j - 1])  # Top-left
                if j < cols - 1:
                    min_cost = min(min_cost, dp[i - 1, j + 1])  # Top-right
                dp[i, j] = cost[i, j] + min_cost

        # Backtrack to find the path of the minimum cost seam
        seam = np.zeros(rows, dtype=np.int32)
        seam[-1] = np.argmin(dp[-1, :])  # Grab the minimum cost from the last row

        for i in range(rows - 2, -1, -1):
            j = seam[i + 1]
            j_from = max(j - 1, 0)
            j_to = min(j + 1, cols - 1) + 1
            seam[i] = j_from + np.argmin(dp[i, j_from:j_to])

        return seam

    def blend_seam(
        self, curr: np.ndarray, nxt: np.ndarray, seam: np.ndarray, blend_width: int = 5
    ) -> np.ndarray:
        blended_image = np.copy(curr)
        rows, cols = curr.shape[:2]

        for i in range(rows):
            j = seam[i]
            for w in range(-blend_width, blend_width + 1):
                if 0 <= j + w < cols:
                    alpha = (blend_width - abs(w)) / blend_width
                    blended_image[i, j + w] = (
                        alpha * curr[i, j + w] + (1 - alpha) * nxt[i, j + w]
                    )
        return blended_image

    def synthesize_texture(self) -> np.ndarray:
        texture = self.random_patch()
        curr_patch = texture
        i = 1

        while texture.shape[1] < self.target_width:
            print(f"Iteration: {i}")
            nxt_patch = self.find_min_ssd_patch(curr_patch)
            seam = self.calc_best_seam(curr_patch, nxt_patch)
            blended_area = self.blend_seam(curr_patch, nxt_patch, seam)
            texture = np.hstack((texture[:, : -self.overlap], blended_area))
            curr_patch = nxt_patch
            i += 1
        return texture[: self.patch_size, : self.target_width]


if __name__ == "__main__":
    quilter = ImageQuilting("assets/TextureSample.jpg")
    res = quilter.synthesize_texture()

    ex = cv2.imread("assets/QuiltingExample.png")
    cv2.imshow("Texture Synthesis", res)
    cv2.imshow("Original Texture", ex)
    cv2.waitKey(0)
