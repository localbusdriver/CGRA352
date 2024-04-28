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
            src, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT
        )
        self.targ_padding = cv2.copyMakeBorder(
            targ, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT
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

        print(f"\n****Initial NNF:\n{self.nnf}\n\n****self.nnd:\n{self.nnd}\n\n")

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

    def propagation(self, x: int, y: int, dir: int) -> None:
        print("\tPropagation")
        # src_h = self.src_padding.shape[0] - self.patch_size + 1
        # src_w = self.src_padding.shape[1] - self.patch_size + 1

        best_y, best_x = self.nnf[y, x]
        best_dist = self.nnd[y, x]

        nx, ny = x + dir, y
        if 0 <= nx < self.nnf.shape[1]:
            candidate_y, candidate_x = self.nnf[ny, nx]

            if 0 <= candidate_x < self.src_padding.shape[1] - self.patch_size:
                candidate_dist = self.calculate_distance(
                    np.array([y, x]), np.array([candidate_y, candidate_x])
                )
                if candidate_dist < best_dist:
                    best_y, best_x = candidate_y, candidate_x
                    best_dist = candidate_dist
            # dist_left = self.nnd[max(x - 1, 0), y]
            # dist_up = self.nnd[x, max(y - 1, 0)]
            # dist_curr = self.nnd[x, y]
            # idx = np.argmin(np.array([dist_curr, dist_left, dist_up]))

            # if idx == 1:
            #     self.nnf[x, y] = self.nnf[max(x - 1, 0), y]
            #     self.nnd[x, y] = self.calculate_distance(
            #         np.array([x, y]), self.nnf[x, y]
            #     )
            # if idx == 2:
            #     self.nnf[x, y] = self.nnf[x, max(y - 1, 0)]
            #     self.nnd[x, y] = self.calculate_distance(
            #         np.array([x, y]), self.nnf[x, y]
            #     )
        nx, ny = x, y + dir
        if 0 <= ny < self.nnf.shape[0]:
            candidate_y, candidate_x = self.nnf[ny, nx]

            if 0 <= candidate_y < self.src_padding.shape[0] - self.patch_size:
                candidate_dist = self.calculate_distance(
                    np.array([y, x]), np.array([candidate_y, candidate_x])
                )
                if candidate_dist < best_dist:
                    best_y, best_x = candidate_y, candidate_x
                    best_dist = candidate_dist

        self.nnf[y, x] = [best_y, best_x]
        self.nnd[y, x] = best_dist

        # else:

        # dist_right = self.nnd[min(x + 1, src_h - 1), y]
        # dist_down = self.nnd[x, min(y + 1, src_w - 1)]
        # dist_curr = self.nnd[x, y]
        # idx = np.argmin(np.array([dist_curr, dist_right, dist_down]))
        # if idx == 1:
        #     self.nnf[x, y] = self.nnf[min(x + 1, src_h - 1), y]
        #     self.nnd[x, y] = self.calculate_distance(
        #         np.array([x, y]), self.nnf[x, y]
        #     )
        # if idx == 2:
        #     self.nnf[x, y] = self.nnf[x, min(y + 1, src_w - 1)]
        #     self.nnd[x, y] = self.calculate_distance(
        #         np.array([x, y]), self.nnf[x, y]
        #     )

    def random_search(self, x: int, y: int) -> None:
        print("\tRandom Search")
        best_dy, best_dx = self.nnf[y, x]
        best_dist = self.nnd[y, x]
        radius = max(
            self.src_padding.shape[0], self.src_padding.shape[1]
        )  # max of height and width

        while radius > 1:
            # Search bounds
            min_dy = max(best_dy - radius, 0)
            max_dy = min(best_dy + radius, self.src_padding.shape[0] - self.patch_size)
            min_dx = max(best_dx - radius, 0)
            max_dx = min(best_dx + radius, self.src_padding.shape[1] - self.patch_size)

            candidate_dy = np.random.randint(min_dy, max_dy)  # Random y
            candidate_dx = np.random.randint(min_dx, max_dx)  # Random x

            dist = self.calculate_distance(
                np.array([y, x]), np.array([candidate_dy, candidate_dx])
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
                        self.propagation(x, y, 1)
                    else:
                        self.propagation(x, y, -1)
                    self.random_search(x, y)

    def reconstruction(self):
        print("\n[INFO] Reconstruction")
        self.src_h, self.src_w = self.source.shape[:2]
        temp = np.zeros_like(self.source)
        for i in range(self.src_h):
            for j in range(self.src_w):
                x, y = self.nnf[i, j]
                temp[i, j, :] = self.source[x, y, :]
        return temp


if "__main__" == __name__:

    source = cv2.imread("./assets/Source.jpg", cv2.IMREAD_COLOR)
    target = cv2.imread("./assets/Target.jpg", cv2.IMREAD_COLOR)

    pm = Patchmatch(source, target, patch_size=7, iterations=5)
    start = time.time()
    pm.run()
    NNF = pm.reconstruction()
    print(f"Time taken: {time.time() - start}")
    cv2.imshow("Final", NNF)
    cv2.waitKey(0)
