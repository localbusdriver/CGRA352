import cv2
import numpy as np


def initialize_target(src: np.ndarray, mask: np.ndarray, shift=(-270, 0)):
    target = np.zeros_like(src)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            if mask[y, x] > 0:
                new_x = x + shift[0]
                if 0 <= new_x < src.shape[1]:
                    target[y, new_x] = src[y, x]
            else:
                target[y, x] = src[y, x]
    return target


def build_gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


class Patchmatch:
    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        patch_size: int = 7,
        iterations: int = 5, nnf=None
    ):
        self.src = source
        self.targ = target
        self.patch_size = patch_size
        self.iters = iterations

        self.pad = patch_size // 2
        self.src_padded = cv2.copyMakeBorder(
            self.src, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT
        )
        self.targ_padded = cv2.copyMakeBorder(
            self.targ, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT
        )

        self.nnf = nnf if nnf else np.zeros(
            (target.shape[0], target.shape[1], 2), dtype=np.int32
        )  # Nearest neighbor field
        self.nnd = np.full(
            (target.shape[0], target.shape[1]), np.inf
        )  # distances (costs)
        if not nnf:
            self.initizalize_nnf()

    def initizalize_nnf(self) -> None:
        h, w = self.nnf.shape[:2]
        for y in range(h):
            for x in range(w):
                dy, dx = np.random.randint(h), np.random.randint(w)
                self.nnf[y, x] = [dy, dx]
                self.nnd[y, x] = self.calculate_dists(
                    np.array([x, y]), np.array([dx, dy])
                )
        print(f"\n****Initial NNF:\n{self.nnf}\n\n****self.distance:\n{self.nnd}\n\n")

    def calculate_dists(self, target: np.ndarray, source: np.ndarray) -> float:  # SSD
        x, y = source
        dx, dy = target
        targ_patch = self.targ_padded[y : y + self.patch_size, x : x + self.patch_size]
        src_patch = self.src_padded[
            dy : dy + self.patch_size, dx : dx + self.patch_size
        ]
        diff = targ_patch - src_patch
        num = np.sum(1 - np.int32(np.isnan(diff)))
        dist = np.sum((np.nan_to_num(diff)) ** 2) / num
        return dist

    def propogate(self, x: int, y: int, dir: int) -> None:
        best_dy, best_dx = self.nnf[y, x]
        best_dist = self.nnd[y, x]

        nx, ny = x + dir, y  # horizontal propogation so only x is modified
        if 0 <= nx < self.nnf.shape[1]:
            candidate_dy, candidate_dx = self.nnf[ny, nx]
            candidate_dx += -dir

            if 0 <= candidate_dx < self.src_padded.shape[1] - self.patch_size:
                dist = self.calculate_dists(
                    np.array([x, y]), np.array([candidate_dx, candidate_dy])
                )
                if dist < best_dist:
                    best_dx, best_dy = candidate_dx, candidate_dy
                    best_dist = dist

        nx, ny = x, y + dir  # Vertical propogation so only y is modified
        if 0 <= ny < self.nnf.shape[0]:
            candidate_dy, candidate_dx = self.nnf[ny, nx]
            candidate_dy += -dir

            if 0 <= candidate_dy < self.src_padded.shape[0] - self.patch_size:
                dist = self.calculate_dists(
                    np.array([x, y]), np.array([candidate_dx, candidate_dy])
                )
                if dist < best_dist:
                    best_dy, best_dx = candidate_dy, candidate_dx
                    best_dist = dist

        self.nnf[y, x] = [best_dy, best_dx]
        self.nnd[y, x] = best_dist

    def random_search(self, x: int, y: int) -> None:
        best_dy, best_dx = self.nnf[y, x]
        best_dist = self.nnd[y, x]
        radius = max(self.src_padded.shape[0], self.src_padded.shape[1])

        while radius > 1:
            min_dy = max(best_dy - radius, 0)
            max_dy = min(best_dy + radius, self.src_padded.shape[0] - self.patch_size)
            min_dx = max(best_dx - radius, 0)
            max_dx = min(best_dx + radius, self.src_padded.shape[1] - self.patch_size)

            candidate_dy = np.random.randint(min_dy, max_dy)
            candidate_dx = np.random.randint(min_dx, max_dx)

            dist = self.calculate_dists(
                np.array([x, y]), np.array([candidate_dx, candidate_dy])
            )
            if dist < best_dist:
                best_dx, best_dy = candidate_dx, candidate_dy
                best_dist = dist
            radius //= 2

        self.nnf[y, x] = [best_dy, best_dx]
        self.nnd[y, x] = best_dist
        
    def reconstruct_image(self):
        h, w = self.src.shape[:2]
        temp = np.zeros_like(self.src)
        for i in range(h):
            for j in range(w):
                dy, dx = self.nnf[i, j]
                temp[i, j] = self.src[dy, dx]
        return temp

    def reconstruct_images(self):
        h, w, c = self.targ_padded.shape
        reconstructed = np.zeros((h, w, c), dtype=np.float32)
        patch_count = np.zeros((h, w), dtype=np.int32)

        for y in range(h):
            for x in range(w):
                dy, dx = self.nnf[y, x]  # distance in y and x direction
                src_y = dy + self.pad  # source y = distance y + padding (patch // 2)
                src_x = dx + self.pad  # source x = distance x + padding (patch // 2)

                # Extract patch from source image
                src_patch = self.src_padded[ 
                    src_y : src_y + self.patch_size,
                    src_x : src_x + self.patch_size,
                ]  # from source_y + patch_size && source_x + patch_size

                # Add patch to reconstructed image
                for py in range(self.patch_size-1):
                    for px in range(self.patch_size-1):
                        if 0 <= y + py - self.pad < h and 0 <= x + px - self.pad < w:
                            reconstructed[y + py - self.pad, x + px - self.pad] += src_patch[py, px]
                            patch_count[y + py - self.pad, x + px - self.pad] += 1

        patch_count[patch_count == 0] = 1  # Prevent division by zero
        reconstructed /= patch_count[:, :, np.newaxis]  # Normalize to get average

        return reconstructed.astype(np.uint8)

    def run(self):
        for iter in range(self.iters):
            print(f"\n[INFO] Iteration {iter}\n")
            for y in range(self.targ.shape[0]):
                for x in range(self.targ.shape[1]):
                    if iter % 2 == 0:
                        self.propogate(x, y, 1)
                    else:
                        self.propogate(x, y, -1)
                    self.random_search(x, y)
        return self.nnf


def upscale_nnf(nnf, scale_factor):
    """Upscale an NNF by a given scale factor"""
    sh, sw = scale_factor * np.array(nnf.shape[:2])
    upscaled_nnf = np.zeros((sh, sw, 2), dtype=np.int32)
    for y in range(upscaled_nnf.shape[0]):
        for x in range(upscaled_nnf.shape[1]):
            upscaled_nnf[y, x] = (
                scale_factor * nnf[y // scale_factor, x // scale_factor]
            )
    return upscaled_nnf


if __name__ == "__main__":
    source_img = cv2.imread("assets/ReshuffleSource.jpg", cv2.IMREAD_COLOR)
    mask = cv2.imread("assets/ReshuffleMask.jpg", cv2.IMREAD_GRAYSCALE)
    target_img = initialize_target(source_img, mask)
    pm = Patchmatch(source_img, target_img)
    nearest_neighbor_field = pm.run()
    # Final reconstruction at the finest level
    final_reconstructed = pm.reconstruct_image()
    cv2.imshow("Final Reconstructed Image", final_reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
