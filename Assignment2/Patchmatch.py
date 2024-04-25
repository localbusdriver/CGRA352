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
        self.src_padding = np.pad(
            src, [(self.pad, self.pad), (self.pad, self.pad), (0, 0)], mode="reflect"
        )

        # Initialize NNF and distance matrices
        self.nnf = np.zeros((self.src_h, self.src_w, 2), dtype=np.int32)  # nnf w/ 0s
        self.distance = np.full(
            (self.src_h, self.src_w), np.inf, dtype=np.float32
        )  # (costs)

        self.initialize_nnf()

    def initialize_nnf(self) -> None:

        random_targ_rows = np.random.randint(
            self.pad, self.targ_h - self.pad, (self.src_h, self.src_w)
        )
        random_targ_cols = np.random.randint(
            self.pad, self.targ_w - self.pad, (self.src_h, self.src_w)
        )

        for i in range(self.src_h):
            for j in range(self.src_w):
                self.nnf[i, j] = [random_targ_rows[i, j], random_targ_cols[i, j]]
                self.distance[i, j] = self.calculate_distance(
                    np.array([i, j]),
                    np.array(
                        [random_targ_rows[i, j], random_targ_cols[i, j]], dtype=np.int32
                    ),
                )  # calculate distance (cost)

        print(
            f"\n****Initial NNF:\n{self.nnf}\n\n****self.distance:\n{self.distance}\n\n"
        )

    def calculate_distance(
        self, rand_src: np.array, rand_targ: np.array
    ) -> float:  # ssd
        half_patch = self.patch_size // 2

        patch_src = self.src_padding[
            rand_src[0] : rand_src[0] + self.patch_size,
            rand_src[1] : rand_src[1] + self.patch_size,
            :,
        ]
        patch_targ = self.target[
            rand_targ[0] - half_patch : rand_targ[0] + half_patch + 1,
            rand_targ[1] - half_patch : rand_targ[1] + half_patch + 1,
            :,
        ]
        diff = patch_targ - patch_src
        num = np.sum(1 - np.int32(np.isnan(diff)))
        dist = np.sum((np.nan_to_num(diff))**2) / num
        return dist
    
    def propagation(self, x: int, y: int, is_odd: bool) -> None:
        src_h = self.src_padding.shape[0] - self.patch_size + 1
        src_w = self.src_padding.shape[1] - self.patch_size + 1

        if is_odd:
            dist_left = self.distance[max(x - 1, 0), y]
            dist_up = self.distance[x, max(y - 1, 0)]
            dist_curr = self.distance[x, y]
            idx = np.argmin(np.array([dist_curr, dist_left, dist_up]))

            if idx == 1:
                self.nnf[x, y] = self.nnf[max(x - 1, 0), y]
                self.distance[x, y] = self.calculate_distance(
                    np.array([x, y]), self.nnf[x, y]
                )
            if idx == 2:
                self.nnf[x, y] = self.nnf[x, max(y - 1, 0)]
                self.distance[x, y] = self.calculate_distance(
                    np.array([x, y]), self.nnf[x, y]
                )
        else:
            dist_right = self.distance[min(x + 1, src_h - 1), y]
            dist_down = self.distance[x, min(y + 1, src_w - 1)]
            dist_curr = self.distance[x, y]
            idx = np.argmin(np.array([dist_curr, dist_right, dist_down]))
            if idx == 1:
                self.nnf[x, y] = self.nnf[min(x + 1, src_h - 1), y]
                self.distance[x, y] = self.calculate_distance(
                    np.array([x, y]), self.nnf[x, y]
                )
            if idx == 2:
                self.nnf[x, y] = self.nnf[x, min(y + 1, src_w - 1)]
                self.distance[x, y] = self.calculate_distance(
                    np.array([x, y]), self.nnf[x, y]
                )
    
    def random_search(self, x: int, y: int, alpha=0.5) -> None:
        i = 4
        search_h = self.targ_h * alpha**i
        search_w = self.targ_w * alpha**i
        targ_x = self.nnf[x, y][0]
        targ_y = self.nnf[x, y][1]

        while search_h > 1 and search_w > 1:
            search_min_r = max(targ_x - search_h, self.pad)
            search_max_r = min(targ_x + search_h, self.targ_h - self.pad)
            rand_targ_x = np.random.randint(search_min_r, search_max_r)

            search_min_c = max(targ_y - search_w, self.pad)
            search_max_c = min(targ_y + search_w, self.targ_w - self.pad)
            rand_targ_y = np.random.randint(search_min_c, search_max_c)

            search_h = self.targ_h * alpha**i
            search_w = self.targ_w * alpha**i
            targ = np.array([rand_targ_x, rand_targ_y])
            cost = self.calculate_distance(np.array([x, y]), targ)

            if cost < self.distance[x, y]:
                self.distance[x, y] = cost
                self.nnf[x, y] = targ
            i+=1
        
    
    def iterations(self):

        self.initialize_nnf()

        for iter in range(1, self.itr + 1):
            if iter % 2 == 0:
                for i in range(self.src_h - 1, -1, -1):
                    for j in range(self.src_w - 1, -1, -1):
                        self.propagation(i, j, False)
                        self.random_search(i, j)
            else:
                for i in range(self.src_h):
                    for j in range(self.src_w):
                        self.propagation(i, j, True)
                        self.random_search(i, j)
            print(f"Iteration: {iter}")

    
    def reconstruction(self):
        self.src_h, self.src_w = self.source.shape[:2]
        temp = np.zeros_like(self.source)
        for i in range(self.src_h):
            for j in range(self.src_w):
                x, y = self.nnf[i, j]
                temp[i, j, :] = self.target[x, y, :]
        return temp


if "__main__" == __name__:

    source = cv2.imread("./assets/Source.jpg", cv2.IMREAD_COLOR)
    target = cv2.imread("./assets/Target.jpg", cv2.IMREAD_COLOR)

    pm = Patchmatch(source, target, patch_size=7, iterations=5)
    start = time.time()
    pm.iterations()
    NNF = pm.reconstruction()
    print(f"Time taken: {time.time() - start}")
    cv2.imshow("Final", NNF)
    cv2.waitKey(0)