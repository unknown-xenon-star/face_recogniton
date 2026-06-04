# Mobile Face Authentication Prototype

This repository contains prototype modules for a mobile identity authentication
system that combines face recognition with liveness detection. The intended
architecture executes both checks in parallel after face detection, reducing
authentication latency while preserving a strict security decision:

```text
authentication = liveness_pass AND identity_match
```

The project currently consists of standalone Python scripts for image
preprocessing, face detection/embedding, LBP texture extraction, LBP-DCT temporal
feature extraction, and a Mask R-CNN segmentation demo.

## System Goal

Traditional authentication pipelines often run liveness detection and face
recognition sequentially:

```text
capture -> preprocess -> detect face -> liveness -> recognition -> decision
```

The target system runs the two expensive verification branches concurrently:

```text
capture -> preprocess -> detect face -> crop/resize
                                      -> liveness branch
                                      -> recognition branch
                                      -> final decision
```

This changes the dominant verification cost from:

```text
liveness_time + recognition_time
```

to:

```text
max(liveness_time, recognition_time)
```

The design target described for this system reduces total authentication time
from approximately `1999 ms` in a sequential pipeline to approximately
`1263 ms` with parallel execution, while reporting `98%` accuracy.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `Image_preprocessing.py` | Demonstrates brightness measurement, Gaussian filtering, manual histogram equalization, grayscale stretching, and visualization. |
| `Face_Detection_Module.py` | Uses OpenCV Haar cascade face detection and MobileNetV2 feature extraction from webcam frames. |
| `lbp.py` | Implements a manual 3x3 Local Binary Pattern transform for a grayscale face image. |
| `dct.py` | Extracts uniform LBP histograms from video frames and applies DCT across the temporal feature matrix. |
| `mobile_auth_pipeline.py` | End-to-end prototype orchestrator for enrollment and parallel authentication. |
| `mask.py` | Runs a real-time Mask R-CNN instance segmentation demo using PyTorch and OpenCV. |
| `face.jpg` | Sample image used by `lbp.py`. |
| `low_brightness.jpg` | Sample image used by `Image_preprocessing.py`. |
| `docs/ARCHITECTURE.md` | Detailed system architecture and parallel execution model. |
| `docs/ALGORITHMS.md` | Technical explanation of the preprocessing, LBP-DCT liveness, SIFT recognition, SVM, and weighted matching approach. |
| `docs/USAGE.md` | Environment setup and script execution guide. |

## Current Prototype Status

Implemented in this repository:

- Image brightness calculation and classical preprocessing.
- Webcam face detection using OpenCV Haar cascades.
- End-to-end enrollment and authentication entrypoint.
- Parallel liveness and recognition execution using `ThreadPoolExecutor`.
- MobileNetV2 feature extraction for detected face crops.
- LBP texture image generation.
- Uniform LBP histogram extraction.
- Temporal DCT feature extraction from a frame sequence.
- Regional SIFT feature extraction with ORB fallback when SIFT is unavailable.
- Weighted regional template matching.
- Real-time Mask R-CNN segmentation demo.

Documented target architecture, not yet fully integrated as one production
pipeline:

- SVM liveness model training. The pipeline can load a pickled SVM model, but no
  trained model is included.
- MobileNet-based production face detector. The integrated prototype currently
  uses OpenCV Haar cascades for local execution.
- Secure production template storage and mobile OS integration.
- Final authentication service/API.

## Requirements

The scripts use Python and common computer vision packages:

- Python 3.10+
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- SciPy
- TensorFlow/Keras
- PyTorch
- torchvision

See [docs/USAGE.md](docs/USAGE.md) for setup and execution examples.

## Quick Start

Enroll an identity template from webcam source `0`:

```bash
python mobile_auth_pipeline.py enroll --source 0 --template templates/default_identity.npz
```

Run parallel authentication against that template:

```bash
python mobile_auth_pipeline.py auth --source 0 --template templates/default_identity.npz
```

Use `--no-preview` when running without OpenCV display windows.

## High-Level Pipeline

1. The mobile camera captures a video stream.
2. The system estimates grayscale brightness and activates supplemental light if
   the face image is below threshold.
3. Frames are preprocessed with histogram equalization, Gaussian filtering, and
   grayscale stretching.
4. A lightweight face detector finds the face bounding box.
5. The face crop is normalized to a fixed size, typically `64x64` for liveness
   feature extraction.
6. The normalized face crop is sent to two branches in parallel:
   - Liveness detection using LBP-DCT features and SVM classification.
   - Face recognition using regional SIFT features and weighted template
     matching.
7. Authentication succeeds only when both branches return positive results.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Algorithms](docs/ALGORITHMS.md)
- [Usage](docs/USAGE.md)

## Security Notes

This repository is a prototype and should not be treated as a production-ready
biometric authentication system. Production use requires model validation,
anti-spoof testing across attack types, secure template storage, on-device data
protection, enrollment workflows, threshold calibration, and privacy review.
