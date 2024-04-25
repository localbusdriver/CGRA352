import numpy as np
import cv2


class Reshuffler:
    def __init__(
        self,
        src: np.ndarray,
        targ: np.ndarray,
        mask: np.ndarray,
        patch_size: int = 7,
        iterations: int = 5,
        shift=(0, 0),
    ):
        self.src = src
        self.targ = targ
        self.mask = mask
        self.patch_size = patch_size
        self.iters = iterations
        self.shift = shift

        self.pad = patch_size // 2
        self.src_h, self.src_w = src.shape[:2]
        self.targ_h, self.targ_w = targ.shape[:2]

        self.src_padded = np.pad(
            src, [(self.pad, self.pad), (self.pad, self.pad), (0, 0)], mode="reflect"
        )
        self.nnf = np.zeros((self.src_h, self.src_w, 2), dtype=np.int32)
        self.distance = np.full((self.src_h, self.src_w), np.inf, dtype=np.float32)

        self.initialize_nnf()

    def initialize_nnf(self) -> None:
        for i in range(self.src_h):
            for j in range(self.src_h):
                x, y = i + self.shift[0], j + self.shift[1]
                x = np.clip(
                    x, 0, self.targ_h - 1
                )  # limits the value of x to be between 0 and targ_h-1
                y = np.clip(
                    y, 0, self.targ_w - 1
                )  # limits the value of y to be between 0 and targ_w-1

                self.nnf[i, j] = [x, y]
                self.distance[i, j] = self.calculate_distance(i, j, x, y)

    def calculate_distance(self, sx, sy, tx, ty) -> float:  # costs
        src_patch = self.src_padded[
            sx : sx + self.patch_size, sy : sy + self.patch_size
        ]
        targ_patch = self.targ[
            tx - self.pad : tx + self.pad + 1, ty - self.pad : ty + self.pad + 1
        ]
        diff = src_patch - targ_patch
        num = np.sum(1 - np.int32(np.isnan(diff)))
        dist = np.sum((np.nan_to_num(diff))**2) / num
        return dist

    def propagate(self):
        pass

    def random_search(self):
        pass

    def reconstruct_image(self):
        reconstruction = np.zeros_like(self.src)
        return reconstruction

    def run(self):
        for i in range(self.iters):
            print(f"Iteration: {i}")
            self.propagate_and_random_search()
        return self.reconstruct_image()


if __name__ == "__main__":
    reshuffle_src = cv2.imread("assets/ReshuffleSource.jpg")
    reshuffle_targ = cv2.imread("assets/Reshuffle.jpg")
    reshuffle_mask = cv2.imread("assets/ReshuffleMask.jpg", 0)
    reshuffler = Reshuffler(
        reshuffle_src, reshuffle_targ, reshuffle_mask, shift=(-270, 0)
    )
    res = reshuffler.run()
    cv2.imshow("Reshuffled Image", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
