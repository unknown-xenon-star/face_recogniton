# Architecture

## Objective

The system is designed for mobile identity authentication where user experience
and attack resistance both matter. The core architectural improvement is to run
liveness detection and face recognition in parallel after a face crop has been
detected and normalized.

Sequential authentication is simple but slow:

```text
camera
  -> preprocessing
  -> face detection
  -> liveness detection
  -> face recognition
  -> decision
```

The target architecture shares the capture, preprocessing, and detection work,
then forks the verification stage:

```text
camera
  -> brightness control
  -> preprocessing
  -> face detection
  -> crop and resize
       -> liveness detection
       -> face recognition
  -> decision fusion
```

## Runtime Pipeline

### 1. Image Acquisition

The mobile terminal captures frames from the front-facing camera. For each frame
or detected face region, the system computes the average grayscale value:

```text
gray_avg = mean(gray_pixels)
```

If `gray_avg` is below a configured threshold, the system estimates the
additional light required and enables the flash or screen illumination. This is
done before expensive feature extraction because low-light input damages both
texture quality and identity feature stability.

### 2. Image Preprocessing

Preprocessing uses classical image operations that are inexpensive enough for
mobile hardware:

- Histogram equalization to improve contrast and redistribute grayscale values.
- Gaussian filtering to reduce sensor noise and small environmental artifacts.
- Grayscale stretching to expand useful intensity ranges with a linear
  transformation.

These steps make downstream features more stable without introducing a heavy
image enhancement model.

### 3. Face Detection

A lightweight detector, such as a MobileNet-family detector, localizes face
bounding boxes. The current prototype also contains an OpenCV Haar cascade demo
in `Face_Detection_Module.py`.

After detection, the selected face region is cropped and resized. The target
liveness branch uses a compact `64x64` face image to reduce memory traffic and
feature extraction cost.

### 4. Parallel Verification

The normalized face crop is dispatched to two independent branches:

```text
normalized face crop
  -> liveness worker
  -> recognition worker
```

The branches do not depend on each other's intermediate values, so they can run
on separate CPU threads, mobile GPU/DSP/NPU resources, or a mixed scheduling
strategy.

The expected timing improvement comes from replacing additive latency:

```text
T_total = T_shared + T_liveness + T_recognition + T_decision
```

with overlapping latency:

```text
T_total = T_shared + max(T_liveness, T_recognition) + T_decision
```

For the described system, this reduces authentication time from approximately
`1999 ms` to `1263 ms`.

### 5. Decision Fusion

The final decision is intentionally strict:

```text
authentication_success = liveness_pass AND identity_match
```

A live but unknown face fails. A recognized but spoofed face also fails. This
keeps the latency benefit of parallel execution without weakening the security
policy.

## Production Component Boundaries

A production implementation should separate the system into these components:

| Component | Responsibility |
| --- | --- |
| `CameraController` | Camera frame acquisition, exposure state, flash/screen illumination control. |
| `Preprocessor` | Brightness estimation, histogram equalization, Gaussian filtering, grayscale stretching. |
| `FaceDetector` | Face localization and crop generation. |
| `LivenessEngine` | LBP extraction, temporal DCT feature fusion, SVM inference. |
| `RecognitionEngine` | Facial region extraction, SIFT descriptors, weighted template matching. |
| `DecisionEngine` | Threshold checks, result fusion, audit metadata. |
| `TemplateStore` | Secure enrollment template storage and retrieval. |

## Mobile Optimization Strategy

The design reduces resource occupation through several choices:

- Shared preprocessing and detection work avoids duplicate frame processing.
- Fixed-size face crops constrain memory and computation.
- LBP-DCT liveness features are lighter than large temporal neural networks.
- SVM inference is inexpensive once trained.
- Regional SIFT limits feature extraction to discriminative facial areas.
- Parallel execution improves hardware utilization and reduces user-visible
  latency.

## Prototype Mapping

The current repository maps to the architecture as follows:

| Target module | Prototype file |
| --- | --- |
| Brightness and preprocessing | `Image_preprocessing.py` |
| Face detection and embedding | `Face_Detection_Module.py` |
| LBP spatial texture extraction | `lbp.py` |
| LBP-DCT temporal feature extraction | `dct.py` |
| End-to-end parallel authentication | `mobile_auth_pipeline.py` |
| Segmentation experiment | `mask.py` |

The integrated orchestrator is implemented in `mobile_auth_pipeline.py`. It is
still a prototype: it uses OpenCV Haar detection instead of a production
MobileNet detector, local `.npz` templates instead of secure biometric storage,
and a heuristic liveness threshold unless a trained SVM model is supplied.
