import cv2 
import numpy as np

class Patchmatch:
    def __init__(
        self,
        src: np.ndarray,
        targ: np.ndarray,
        patch_size: int = 7,
        iterations: int = 5,
    ):
        self.patch_size = patch_size
        self.itr = iterations
        self.source = src
        self.target = targ

        self.pad = patch_size // 2

        # Widths and Heights
        self.src_h, self.src_w = src.shape[:2]
        self.targ_h, self.targ_w = targ.shape[:2]

        # Initialize the source image with padding with 'reflect'
        self.src_padding = cv2.copyMakeBorder(
            src, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_CONSTANT
        )
        self.targ_padding = cv2.copyMakeBorder(
            targ, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_CONSTANT
        )

        # Initialize NNF and distance matrices
        self.nnf = np.zeros((self.src_h, self.src_w, 2), dtype=np.int32)  # nnf w/ 0s
        self.nnd = np.full((self.src_h, self.src_w), np.inf)  # (costs)

        self.initialize_nnf()

    def initialize_nnf(self) -> None:
        print("\n[INFO] Initializing NNF")
        h, w = self.nnf.shape[:2]
        for y in range(h):
            for x in range(w):
                dy, dx = np.random.randint(h), np.random.randint(w)
                self.nnf[y, x] = [dy, dx]
                self.nnd[y, x] = self.calculate_distance(
                    np.array([y, x]),
                    np.array([dy, dx]),
                )  # calculate distance (cost)
        # print(f"\n****Initial NNF:\n{self.nnf}\n\n****self.nnd:\n{self.nnd}\n\n")

    def calculate_distance(
        self, rand_src: np.array, rand_targ: np.array
    ) -> float:  # ssd

        patch_src = self.src_padding[
            rand_src[0] : rand_src[0] + self.patch_size,
            rand_src[1] : rand_src[1] + self.patch_size,
        ]

        patch_targ = self.targ_padding[
            rand_targ[0] : rand_targ[0] + self.patch_size,
            rand_targ[1] : rand_targ[1] + self.patch_size,
        ]

        # dist = np.sum((patch_targ - patch_src) ** 2)
        diff = patch_targ - patch_src  # Difference between patches
        num = np.sum(1 - np.int32(np.isnan(diff)))  # Number of valid pixels
        dist = np.sum((np.nan_to_num(diff)) ** 2) / num  # Calculate distance
        return dist
      
    def getnnf(self):
        return self.nnf


src, targ = cv2.imread("assets/Source.jpg"), cv2.imread("assets/Target.jpg")
pm = Patchmatch(src, targ)
nnf = pm.getnnf()

print(f"Source shape: {src.shape}\nTarget shape: {targ.shape}\nNNF shape: {nnf.shape}")
