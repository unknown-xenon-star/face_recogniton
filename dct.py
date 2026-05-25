import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct

# -----------------------------
# PARAMETERS
# -----------------------------
RADIUS = 1
N_POINTS = 8 * RADIUS
METHOD = 'uniform'

IMG_SIZE = (64, 64)
DCT_COMPONENTS = 20

# -----------------------------
# LBP FEATURE EXTRACTION
# -----------------------------
def extract_lbp(image):

    image = cv2.resize(image, IMG_SIZE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(
        gray,
        N_POINTS,
        RADIUS,
        METHOD
    )

    # 59-bin histogram
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 60),
        range=(0, 59)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist, lbp


# -----------------------------
# DCT FEATURE EXTRACTION
# -----------------------------
def apply_dct(feature_matrix):

    dct_features = []

    for col in range(feature_matrix.shape[1]):

        signal = feature_matrix[:, col]

        coeff = dct(signal, norm='ortho')

        coeff = coeff[:DCT_COMPONENTS]

        dct_features.extend(coeff)

    return np.array(dct_features)

def main():
    # -----------------------------
    # CAMERA
    # -----------------------------
    cap = cv2.VideoCapture(0)

    lbp_vectors = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Extract LBP
        hist, lbp_image = extract_lbp(frame)

        lbp_vectors.append(hist)

        # Normalize LBP image for display
        lbp_display = cv2.normalize(
            lbp_image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Show original frame
        cv2.imshow("Original", frame)

        # Show LBP texture image
        cv2.imshow("LBP", lbp_display)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # -----------------------------
    # CREATE FEATURE MATRIX
    # -----------------------------
    feature_matrix = np.array(lbp_vectors)

    print("LBP Feature Matrix Shape:", feature_matrix.shape)

    # -----------------------------
    # APPLY DCT
    # -----------------------------
    final_features = apply_dct(feature_matrix)

    print("Final DCT Feature Shape:", final_features.shape)

    print(final_features)

if __name__ == "__main__":
    main()