import cv2
import numpy as np
import time


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
        self.nnf = np.zeros((self.targ_h, self.targ_w, 2), dtype=np.int32)  # nnf w/ 0s
        self.nnd = np.full((self.targ_h, self.targ_w), np.inf)  # (costs)

        self.initialize_nnf()

    def initialize_nnf(self) -> None:
        print("\n[INFO] Initializing NNF")
        h, w = self.nnf.shape[:2]
        for y in range(h):
            for x in range(w):
                dy, dx = np.random.randint(h), np.random.randint(w)
                self.nnf[y, x] = [dy, dx]
                self.nnd[y, x] = self.calculate_distance(
                    np.array([dy, dx]),
                    np.array([y, x]),
                )  # calculate distance (cost)
        # print(f"\n****Initial NNF:\n{self.nnf}\n\n****self.nnd:\n{self.nnd}\n\n")

    def calculate_distance(self, src: np.array, targ: np.array) -> float:  # ssd
        src_y, src_x = src[0], src[1]
        patch_src = self.src_padding[
            src_y : src_y + self.patch_size,
            src_x : src_x + self.patch_size,
            :,
        ]

        targ_y, targ_x = targ[0], targ[1]
        patch_targ = self.targ_padding[
            targ_y : targ_y + self.patch_size,
            targ_x : targ_x + self.patch_size,
            :,
        ]
        diff = patch_targ - patch_src  # Difference between patches
        num = np.sum(1 - np.int32(np.isnan(diff)))  # Number of valid pixels
        dist = np.sum((np.nan_to_num(diff)) ** 2) / num  # Calculate distance
        return dist

    def propagation(self, x: int, y: int, dir: int) -> None:
        # print("\tPropagation")

        best_y, best_x = self.nnf[y, x]
        best_dist = self.nnd[y, x]

        nx, ny = x + dir, y
        if 0 <= nx < self.nnf.shape[1]:
            candidate_y, candidate_x = self.nnf[ny, nx]
            # candidate_x += dir

            if 0 <= candidate_x < self.target.shape[1]:
                candidate_dist = self.calculate_distance(
                    np.array([candidate_y, candidate_x]), np.array([y, x])
                )
                if candidate_dist < best_dist:
                    best_y, best_x = candidate_y, candidate_x
                    best_dist = candidate_dist

        nx, ny = x, y + dir
        if 0 <= ny < self.nnf.shape[0]:
            candidate_y, candidate_x = self.nnf[ny, nx]
            # candidate_y += dir

            if 0 <= candidate_y < self.target.shape[0]:
                candidate_dist = self.calculate_distance(
                    np.array([y, x]), np.array([candidate_y, candidate_x])
                )
                if candidate_dist < best_dist:
                    best_y, best_x = candidate_y, candidate_x
                    best_dist = candidate_dist

        self.nnf[y, x] = [best_y, best_x]
        self.nnd[y, x] = best_dist

    def random_search(self, x: int, y: int) -> None:
        # print("\tRandom Search")
        best_dy, best_dx = self.nnf[y, x]
        best_dist = self.nnd[y, x]
        radius = max(
            self.target.shape[0], self.target.shape[1]
        )  # max of height and width

        while radius > 1:
            # Search bounds
            min_dy = max(best_dy - radius, 0)
            max_dy = min(best_dy + radius, self.target.shape[0])
            min_dx = max(best_dx - radius, 0)
            max_dx = min(best_dx + radius, self.target.shape[1])

            candidate_dy = np.random.randint(min_dy, max_dy)  # Random y
            candidate_dx = np.random.randint(min_dx, max_dx)  # Random x

            dist = self.calculate_distance(
                np.array([candidate_dy, candidate_dx]), np.array([y, x])
            )
            if dist < best_dist:
                best_dx, best_dy = candidate_dx, candidate_dy
                best_dist = dist
            radius //= 2  # Decrease radius to find better match

        self.nnf[y, x] = [best_dy, best_dx]  # Update best match
        self.nnd[y, x] = best_dist  # Update best distance

    def run(self):
        for iter in range(1, self.itr + 1):
            print(f"\n[INFO] Iteration {iter}\n")
            for y in range(self.target.shape[0]):
                for x in range(self.target.shape[1]):
                    # Check all pixels
                    if iter % 2 == 0:
                        self.propagation(x, y, -1)
                    else:
                        self.propagation(x, y, 1)
                    self.random_search(x, y)
            res, recon = self.reconstruction()
            # cv2.imshow(f"Result + {iter}", res)
            # cv2.imshow(f"Reconstructed Image + {iter}", recon)
            # cv2.waitKey(0)

    def reconstruction(self):
        print("\n[INFO] Reconstruction")
        temp = np.zeros((self.nnf.shape[0], self.nnf.shape[1], 3), dtype=np.uint8)
        for i in range(self.nnf.shape[0]):
            for j in range(self.nnf.shape[1]):
                y, x = self.nnf[i, j]
                if (
                    y < 0
                    or x < 0
                    or y >= self.source.shape[0]
                    or x >= self.source.shape[1]
                ):
                    print("Coordinates out of bounds")

                r = int(x * 255.0 / self.source.shape[1])
                g = int(y * 255.0 / self.source.shape[0])
                b = 255.0 - max(r, g)
                temp[i, j] = [b, g, r]

        reconstructed = np.zeros(self.target.shape, dtype=np.uint8)
        for y in range(self.target.shape[0]):
            for x in range(self.target.shape[1]):
                dy, dx = self.nnf[y, x]  # distance in x and y
                reconstructed[y, x, :] = self.source[
                    dy, dx, :
                ]  # get the pixel from source image
        return temp, reconstructed


if "__main__" == __name__:
    source = cv2.imread("./assets/Source.jpg", cv2.IMREAD_COLOR)
    target = cv2.imread("./assets/Target.jpg", cv2.IMREAD_COLOR)

    pm = Patchmatch(source, target, patch_size=7, iterations=5)
    start = time.time()
    pm.run()
    res, reconstructed = pm.reconstruction()
    print(f"Time taken: {time.time() - start}")
    cv2.imshow("Result NNF", res)
    cv2.imshow("Final", reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
