import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------

img = cv2.imread("low_brightness.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found")


# ------------------------------------------------
# BRIGHTNESS FUNCTION
# ------------------------------------------------

def brightness(img: np.ndarray) -> float:
    return float(img.mean())


# ------------------------------------------------
# GAUSSIAN FILTERING
# ------------------------------------------------

def gaussian_filter(img: np.ndarray,
                    kernel_size=(5, 5),
                    sigma=1.0) -> np.ndarray:

    blurred = cv2.GaussianBlur(
        img,
        kernel_size,
        sigma
    )

    return blurred


# ------------------------------------------------
# HISTOGRAM FUNCTION
# ------------------------------------------------

def calculate_histogram(img: np.ndarray) -> np.ndarray:

    hist = np.zeros(256, dtype=int)

    for pixel in img.flatten():
        hist[pixel] += 1

    return hist


# ------------------------------------------------
# PROBABILITY FUNCTION
# p(rk) = n(rk) / N
# ------------------------------------------------

def calculate_probability(hist: np.ndarray,
                          total_pixels: int) -> np.ndarray:

    return hist / total_pixels


# ------------------------------------------------
# CDF FUNCTION
# ------------------------------------------------

def calculate_cdf(prob: np.ndarray) -> np.ndarray:

    return np.cumsum(prob)


# ------------------------------------------------
# HISTOGRAM EQUALIZATION
# Sk = (L-1) * CDF
# ------------------------------------------------

def histogram_equalization(img: np.ndarray,
                           cdf: np.ndarray) -> np.ndarray:

    L = 256

    equalized_values = np.floor(
        (L - 1) * cdf
    ).astype(np.uint8)

    equalized_img = equalized_values[img]

    return equalized_img


# ------------------------------------------------
# GRAYSCALE STRETCHING
# Linear Contrast Stretching
# ------------------------------------------------

def grayscale_stretch(img: np.ndarray) -> np.ndarray:

    r_min = np.min(img)
    r_max = np.max(img)

    stretched = (
        (img - r_min) /
        (r_max - r_min)
    ) * 255

    return stretched.astype(np.uint8)


# ------------------------------------------------
# DISPLAY FUNCTION
# ------------------------------------------------

def display_results(original,
                    gaussian,
                    stretched,
                    equalized):

    plt.figure(figsize=(14, 10))

    # Original Image
    plt.subplot(4, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # Original Histogram
    plt.subplot(4, 2, 2)
    plt.hist(original.flatten(),
             bins=256,
             range=[0, 256])
    plt.title("Original Histogram")

    # Gaussian Image
    plt.subplot(4, 2, 3)
    plt.imshow(gaussian, cmap='gray')
    plt.title("Gaussian Filtered")
    plt.axis("off")

    # Gaussian Histogram
    plt.subplot(4, 2, 4)
    plt.hist(gaussian.flatten(),
             bins=256,
             range=[0, 256])
    plt.title("Gaussian Histogram")

    # Stretched Image
    plt.subplot(4, 2, 5)
    plt.imshow(stretched, cmap='gray')
    plt.title("Grayscale Stretched")
    plt.axis("off")

    # Stretched Histogram
    plt.subplot(4, 2, 6)
    plt.hist(stretched.flatten(),
             bins=256,
             range=[0, 256])
    plt.title("Stretched Histogram")

    # Equalized Image
    plt.subplot(4, 2, 7)
    plt.imshow(equalized, cmap='gray')
    plt.title("Histogram Equalized")
    plt.axis("off")

    # Equalized Histogram
    plt.subplot(4, 2, 8)
    plt.hist(equalized.flatten(),
             bins=256,
             range=[0, 256])
    plt.title("Equalized Histogram")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

print("Original Brightness:",
      brightness(img))


# STEP 1: Gaussian Filtering
gaussian_img = gaussian_filter(img)


# STEP 2: Histogram
hist = calculate_histogram(gaussian_img)


# STEP 3: Probability Distribution
N = gaussian_img.shape[0] * gaussian_img.shape[1]

prob = calculate_probability(hist, N)


# STEP 4: CDF
cdf = calculate_cdf(prob)


# STEP 5: Histogram Equalization
equalized_img = histogram_equalization(
    gaussian_img,
    cdf
)


# STEP 6: Grayscale Stretching
stretched_img = grayscale_stretch(
    gaussian_img
)


# DISPLAY
display_results(
    img,
    gaussian_img,
    stretched_img,
    equalized_img
)