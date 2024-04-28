import cv2
import numpy as np
import time


def initialize_target(src: np.ndarray, mask: np.ndarray, shift=(-270, 0)):
    print("\n[INFO] Initializing target image\n")
    target = np.zeros_like(src)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            if mask[y, x] > 0:  # If mask is not black
                new_x = x + shift[0]  # Shift x by -270
                if 0 <= new_x < src.shape[1]:
                    target[y, new_x] = src[y, x]
            else:
                target[y, x] = src[y, x]
    return target


def build_gaussian_pyramid(image: np.ndarray, k_levels=4) -> list[np.ndarray]:
    print("\n[INFO] Building Gaussian Pyramid\n")
    pyramid = [image]
    for _ in range(0, k_levels):
        image = cv2.pyrDown(image)  # Down sample the image
        pyramid.append(image)
    return pyramid


class Patchmatch:
    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        patch_size: int = 7,
        iterations: int = 5,
    ):
        self.patch_size = patch_size
        self.iters = iterations
        self.src = source
        self.targ = target

        self.pad = patch_size // 2
        self.src_padded = cv2.copyMakeBorder(
            source, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT
        )  # Add border
        self.targ_padded = cv2.copyMakeBorder(
            target, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT
        )  # Add border

        self.nnf = np.zeros(
            (target.shape[0], target.shape[1], 2), dtype=np.int32
        )  # Nearest neighbor field
        self.nnd = np.full(
            (target.shape[0], target.shape[1]), np.inf
        )  # distances (costs)
        self.initizalize_nnf()

    def set_sourceTarget(self, source: np.ndarray, target: np.ndarray) -> None:
        self.src = source
        self.targ = target

    def initizalize_nnf(self) -> None:
        print("\n[INFO] Initializing NNF")
        h, w = self.nnf.shape[:2]
        for y in range(h):
            for x in range(w):
                dy, dx = np.random.randint(h), np.random.randint(w)
                self.nnf[y, x] = [dy, dx]
                self.nnd[y, x] = self.calculate_dists(
                    np.array([y, x]), np.array([dy, dx])
                )
        print(f"\n****Initial NNF:\n{self.nnf}\n\n****self.distance:\n{self.nnd}\n\n")

    def calculate_dists(self, source: np.ndarray, target: np.ndarray) -> float:
        """Calculate cost between source and target patches"""
        y, x = source  # source x and y
        ty, tx = target  # target x and y
        src_patch = self.src_padded[
            y : y + self.patch_size, x : x + self.patch_size
        ]  # Desired patch
        targ_patch = self.targ_padded[
            ty : ty + self.patch_size, tx : tx + self.patch_size
        ]
        diff = targ_patch - src_patch  # Difference between patches
        num = np.sum(1 - np.int32(np.isnan(diff)))  # Number of valid pixels
        dist = np.sum((np.nan_to_num(diff)) ** 2) / num  # Calculate distance
        return dist

    def propogate(self, x: int, y: int, dir: int) -> None:
        print(f"\tPropagation")
        best_dy, best_dx = self.nnf[y, x]
        best_dist = self.nnd[y, x]

        nx, ny = x + dir, y  # horizontal propogation so only x is modified
        if 0 <= nx < self.nnf.shape[1]:
            # if nx between 0 and width of nnf
            candidate_dy, candidate_dx = self.nnf[ny, nx]  # candidate x and y
            candidate_dx += -dir  # candidate x is modified b/c horizontal propogation

            if (
                0 <= candidate_dx < self.src_padded.shape[1] - self.patch_size
            ):  # ensure candidate_dx is within bounds
                dist = self.calculate_dists(
                    np.array([y, x]), np.array([candidate_dy, candidate_dx])
                )
                if dist < best_dist:  # if new best dist
                    best_dx, best_dy = candidate_dx, candidate_dy
                    best_dist = dist

        nx, ny = x, y + dir  # Vertical propogation so only y is modified
        if 0 <= ny < self.nnf.shape[0]:
            # if ny between 0 and width of nnf
            candidate_dy, candidate_dx = self.nnf[ny, nx]
            candidate_dy += -dir  # candidate y is modified b/c vertical propogation

            if 0 <= candidate_dy < self.src_padded.shape[0] - self.patch_size:
                dist = self.calculate_dists(
                    np.array([y, x]), np.array([candidate_dy, candidate_dx])
                )
                if dist < best_dist:
                    best_dy, best_dx = candidate_dy, candidate_dx
                    best_dist = dist

        self.nnf[y, x] = [best_dy, best_dx]
        self.nnd[y, x] = best_dist

    def random_search(self, x: int, y: int) -> None:
        print(f"\tRandom Search")
        best_dy, best_dx = self.nnf[y, x]
        best_dist = self.nnd[y, x]
        radius = max(
            self.src_padded.shape[0], self.src_padded.shape[1]
        )  # max of height and width

        while radius > 1:
            # Search bounds
            min_dy = max(best_dy - radius, 0)
            max_dy = min(best_dy + radius, self.src_padded.shape[0] - self.patch_size)
            min_dx = max(best_dx - radius, 0)
            max_dx = min(best_dx + radius, self.src_padded.shape[1] - self.patch_size)

            candidate_dy = np.random.randint(min_dy, max_dy)  # Random y
            candidate_dx = np.random.randint(min_dx, max_dx)  # Random x

            dist = self.calculate_dists(
                np.array([y, x]), np.array([candidate_dy, candidate_dx])
            )
            if dist < best_dist:
                best_dx, best_dy = candidate_dx, candidate_dy
                best_dist = dist
            radius //= 2  # Decrease radius to find better match

        self.nnf[y, x] = [best_dy, best_dx]  # Update best match
        self.nnd[y, x] = best_dist  # Update best distance

    def run(self, iterations=0):
        iters = iterations if iterations != 0 else self.iters
        for iter in range(1, iters + 1):
            print(f"\n[INFO] Iteration {iter}")
            for y in range(self.src.shape[0]):
                for x in range(self.src.shape[1]):
                    # Check all pixels
                    if iter % 2 == 0:  # Forward propogation
                        self.propogate(x, y, 1)
                    else:  # Backward propogation
                        self.propogate(x, y, -1)
                    self.random_search(x, y)
        return self.nnf

    def reconstruct_image(self):
        print("\n[INFO] Reconstructing Image")
        h, w = self.src.shape[:2]
        temp = np.zeros_like(self.src)
        for y in range(h):
            for x in range(w):
                dy, dx = self.nnf[y, x]
                temp[y, x, :] = self.src[dy, dx, :]
        return temp

    def reconstruct_images(self):
        print("\n[INFO] Reconstructing Images")
        h, w, c = self.src.shape
        reconstructed = np.zeros((h, w, c), dtype=np.float32)
        patch_count = np.zeros((h, w), dtype=np.int32)

        for y in range(h):
            for x in range(w):
                dy, dx = self.nnf[y, x]  # y and x neighbor
                src_y = dy + self.pad  # source y = distance y + padding (patch // 2)
                src_x = dx + self.pad  # source x = distance x + padding (patch // 2)

                # Extract patch from source image
                src_patch = self.src_padded[
                    src_y : src_y + self.patch_size,
                    src_x : src_x + self.patch_size,
                ]  # from source_y + patch_size && source_x + patch_size

                # Add patch to reconstructed image
                for py in range(self.patch_size - 1):
                    for px in range(self.patch_size - 1):
                        if 0 <= y + py - self.pad < h and 0 <= x + px - self.pad < w:
                            # if y + py - pad is between 0 and height && x + px - pad is between 0 and width
                            reconstructed[
                                y + py - self.pad, x + px - self.pad
                            ] += src_patch[py, px]
                            patch_count[y + py - self.pad, x + px - self.pad] += 1

        patch_count[patch_count == 0] = 1  # Prevent division by zero
        reconstructed /= patch_count[:, :, np.newaxis]  # Normalize to get average

        return reconstructed.astype(np.uint8)


def show_all_results(imgs: list[np.ndarray]) -> None:
    print("\n[INFO] Displaying all results of pyramid\n")
    for i in range(len(imgs)):
        cv2.imshow(f"{i+1}th Level Result", imgs[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    # Read images and initialize target image
    source_img = cv2.imread("assets/ReshuffleSource.jpg", cv2.IMREAD_COLOR)
    mask = cv2.imread("assets/ReshuffleMask.jpg", cv2.IMREAD_GRAYSCALE)
    target_img = initialize_target(source_img, mask)

    k_levels = 4  # k-Number of levels in the pyramid
    src_pyramid, targ_pyramid = build_gaussian_pyramid(
        source_img, k_levels=k_levels
    ), build_gaussian_pyramid(target_img, k_levels=k_levels)

    """ Patchmatch START """
    pm = Patchmatch(source_img, target_img)

    # For each level
    res_of_levels = []  # : list[list[np.ndarray]]
    for k in range(k_levels - 1, 0, -1):
        # iterate backwards b/c coarsest level is at the end
        print(f"\n\n[INFO] Running patchmatch on Level {k_levels-k}\n")

        source = src_pyramid[k]
        target = targ_pyramid[k]

        pm.set_sourceTarget(source, target)
        
        pm.run()

        kth_res = pm.reconstruct_image()
        res_of_levels.append(kth_res)

    """ Final Reconstructed Image """
    pm.set_sourceTarget(source_img, target_img)
    final_reconstructed = pm.reconstruct_images()

    show_all_results(res_of_levels)
    cv2.imshow("Final Reconstructed Image", final_reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
