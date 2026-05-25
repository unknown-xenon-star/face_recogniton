import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------

img = cv2.imread("face.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found")


# ------------------------------------------------
# LBP FUNCTION
# ------------------------------------------------

def lbp(image: np.ndarray) -> np.ndarray:

    height, width = image.shape

    # Output image
    lbp_image = np.zeros((height, width),
                         dtype=np.uint8)

    # Ignore border pixels
    for y in range(1, height - 1):
        for x in range(1, width - 1):

            center = image[y, x]

            binary_values = []

            # Clockwise neighbors
            neighbors = [
                image[y-1, x-1],  # top-left
                image[y-1, x],    # top
                image[y-1, x+1],  # top-right
                image[y, x+1],    # right
                image[y+1, x+1],  # bottom-right
                image[y+1, x],    # bottom
                image[y+1, x-1],  # bottom-left
                image[y, x-1]     # left
            ]

            # Threshold neighbors
            for pixel in neighbors:

                if pixel >= center:
                    binary_values.append(1)
                else:
                    binary_values.append(0)

            # Convert binary to decimal
            lbp_value = 0

            for i in range(8):
                lbp_value += (
                    binary_values[i] * (2 ** i)
                )

            lbp_image[y, x] = lbp_value

    return lbp_image


# ------------------------------------------------
# CALCULATE LBP
# ------------------------------------------------

lbp_img = lbp(img)


# ------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------

plt.figure(figsize=(12, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# LBP image
plt.subplot(1, 2, 2)
plt.imshow(lbp_img, cmap='gray')
plt.title("LBP Image")
plt.axis("off")

plt.tight_layout()
plt.show()