import cv2
import numpy as np


class PatchMatch:
    def __init__(
        self,
        source_img: np.ndarray,
        mask_img: np.ndarray,
        patch_size: int = 7,
        iterations: int = 5,
    ):
        self.patch_size = patch_size
        self.itr = iterations
        self.src = source_img
        self.mask = mask_img

        self.w, self.h = source_img.shape[:2]

        self.pad = patch_size // 2
        self.src_padding = np.pad(
            source_img,
            [(self.pad, self.pad), (self.pad, self.pad), (0, 0)],
            mode="reflect",
        )
        # Initialize NNF and distance matrices
        self.nnf = np.zeros((self.src_h, self.src_w, 2), dtype=np.int32)  # nnf w/ 0s
        self.nnd = np.full(
            (self.src_h, self.src_w), np.inf, dtype=np.float32
        )  # (costs)

        self.initialize_nnf()

    def initialize_nnf(self) -> None:
        h, w = self.nnf.shape[:2]
        for y in range(h):
            for x in range(w):
                dy, dx = np.random.randint(h), np.random.randint(w)
                self.nnf[y, x] = [dy, dx]
                self.nnd[y, x] = self.calculate_dists(
                    np.array([x, y]), np.array([dx, dy])
                )
        print(f"\n****Initial NNF:\n{self.nnf}\n\n****self.distance:\n{self.nnd}\n\n")

    def propagate_and_random_search(self, levels):
        # Implement the PatchMatch propagation and random search
        pass

    def reconstruct_image(self):
        # Reconstruct the image from the NNF
        pass


def main():
    source_img = cv2.imread("assets/ToInpaint.jpg", cv2.IMREAD_COLOR)
    mask_img = cv2.imread("assets/ToInpaint_Mask.jpg", cv2.IMREAD_GRAYSCALE)
    nnf = initialize_nnf(source_img, mask_img)
    propagate_and_random_search(nnf, source_img, levels=5)  # Example: 5 pyramid levels
    result_img = reconstruct_image(nnf, source_img)
    cv2.imshow("Inpainted Image", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
