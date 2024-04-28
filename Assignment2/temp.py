import cv2

def build_gaussian_pyramid(image, levels):
    pyramid = [image]  # Start with the original image as the first level of the pyramid.
    for _ in range(1, levels):
        image = cv2.pyrDown(image)  # Apply Gaussian filtering and downsampling
        pyramid.append(image)
    return pyramid

image_path = 'assets/ReshuffleSource.jpg'
original_image = cv2.imread(image_path)

# Specify the number of levels in the pyramid
levels = 4  # Including the original image

# Generate the Gaussian pyramid
pyramid = build_gaussian_pyramid(original_image, levels)

# Optionally, display the pyramid images (requires cv2.imshow or similar)
for i, level_image in enumerate(pyramid):
    cv2.imshow(f'Level {i}', level_image)
    cv2.waitKey(0)
cv2.destroyAllWindows()