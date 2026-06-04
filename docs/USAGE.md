# Usage

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the packages used by the prototype scripts:

```bash
pip install numpy opencv-python matplotlib scikit-image scipy tensorflow torch torchvision
```

Depending on your platform, TensorFlow and PyTorch installation may require
platform-specific wheels. For GPU support, install builds compatible with your
CUDA or accelerator runtime.

## Running The Scripts

Run commands from the repository root.

### End-To-End Parallel Authentication

Create an enrolled identity template:

```bash
python mobile_auth_pipeline.py enroll --source 0 --template templates/default_identity.npz
```

Authenticate against the saved template:

```bash
python mobile_auth_pipeline.py auth --source 0 --template templates/default_identity.npz
```

The authentication command collects a short sequence of detected face crops,
runs LBP-DCT liveness and regional SIFT/ORB recognition concurrently, and prints
the fused decision:

```text
Authenticated: True/False
Parallel branch time: ...
Liveness: ...
Recognition: ...
Region scores:
```

Useful options:

```bash
python mobile_auth_pipeline.py auth --frames 32 --no-preview
python mobile_auth_pipeline.py auth --svm-model models/liveness_svm.pkl
python mobile_auth_pipeline.py auth --liveness-threshold 0.02 --recognition-threshold 0.18
```

When `--svm-model` is omitted, the liveness branch uses a demo temporal-energy
threshold over DCT coefficients. A production system should replace this with a
trained and calibrated SVM.

### Image Preprocessing Demo

```bash
python Image_preprocessing.py
```

This script loads `low_brightness.jpg`, computes the original brightness,
applies Gaussian filtering, histogram equalization, and grayscale stretching,
then displays the resulting images and histograms.

### Manual LBP Demo

```bash
python lbp.py
```

This script loads `face.jpg`, computes a manual 3x3 LBP texture image, and
displays the original and transformed images.

### LBP-DCT Video Feature Demo

```bash
python dct.py
```

This script opens webcam source `0`, extracts uniform LBP histograms from each
frame, displays the live LBP visualization, and applies DCT to the collected
feature matrix after quitting with `q`.

### Face Detection And MobileNetV2 Embedding Demo

```bash
python Face_Detection_Module.py
```

This script opens webcam source `0`, detects faces using an OpenCV Haar cascade,
extracts a MobileNetV2 embedding for each detected face crop, displays the
embedding shape, and prints the first five embedding values.

### Mask R-CNN Segmentation Demo

```bash
python mask.py
```

This script opens webcam source `0`, runs a torchvision Mask R-CNN model, and
renders detected instances with masks and labels. It will download pretrained
weights the first time it runs if they are not already cached.

Configuration values are set near the top of `mask.py`:

```python
SOURCE = 0
SCORE_THRESHOLD = 0.6
MASK_THRESHOLD = 0.5
MAX_DETECTIONS = 10
RESIZE_WIDTH = 960
FORCE_CPU = False
```

## Expected Inputs

- `face.jpg` is required by `lbp.py`.
- `low_brightness.jpg` is required by `Image_preprocessing.py`.
- Webcam access is required by `Face_Detection_Module.py`, `dct.py`, and
  `mask.py`.

## Common Issues

### Webcam Cannot Be Opened

Check that another application is not using the camera. If your webcam is not
device `0`, update the script's source index.

### GUI Windows Do Not Appear

OpenCV display windows require a desktop session. Running over SSH or in a
headless container may require a virtual display or saving outputs to files
instead.

### Model Download Fails

`Face_Detection_Module.py` and `mask.py` use pretrained model weights. The first
run may require network access and enough disk space for local model caches.

## Development Notes

The current repository does not yet expose a single end-to-end authentication
service/API. To evolve the prototype into the documented production
architecture, implement these pieces next:

- A trained SVM model for LBP-DCT liveness features.
- A MobileNet-based detector or mobile-native face detector.
- Secure encrypted template storage.
- Landmark-based facial region extraction instead of fixed proportional crops.
- Threshold calibration and evaluation scripts.
