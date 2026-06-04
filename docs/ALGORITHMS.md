# Algorithms

## Brightness Estimation

Brightness is estimated from the average grayscale value of the captured frame
or face region:

```text
B = (1 / N) * sum(pixel_i)
```

If `B` is below a standard threshold, the mobile terminal can enable
supplemental illumination. This improves downstream feature stability before the
system spends compute on liveness or identity matching.

## Image Preprocessing

### Histogram Equalization

Histogram equalization redistributes grayscale values to improve global
contrast. The prototype computes:

```text
p(r_k) = n_k / N
CDF(k) = sum(p(r_i)), i <= k
s_k = floor((L - 1) * CDF(k))
```

where `L = 256` for 8-bit grayscale images.

### Gaussian Filtering

Gaussian filtering smooths small noise while preserving the broader facial
structure needed by later stages. The prototype uses OpenCV Gaussian blur with a
default `5x5` kernel and `sigma = 1.0`.

### Grayscale Stretching

Linear grayscale stretching expands the observed intensity range:

```text
output = ((input - min) / (max - min)) * 255
```

This improves contrast when the input occupies only a narrow part of the
available grayscale range.

## Face Detection

The target design uses a lightweight neural detector, such as a MobileNet-based
architecture, to keep inference practical on mobile devices. The detector
returns face bounding boxes, and each selected face crop is resized to a fixed
shape before verification.

The current prototype uses OpenCV Haar cascades for detection and MobileNetV2 as
a feature extractor in `Face_Detection_Module.py`.

## Liveness Detection

### LBP Spatial Texture Features

Local Binary Pattern extracts local texture by comparing each pixel with its
`3x3` neighborhood. For each neighbor:

```text
neighbor >= center -> 1
neighbor < center  -> 0
```

The eight binary comparisons form an 8-bit code. A histogram of these codes
represents local facial texture.

The target method uses uniform LBP, producing a compact `1x59` feature vector
for each `64x64` face image. This is useful for liveness detection because real
skin and spoof media often differ in micro-texture, display artifacts,
compression traces, and reflected illumination.

### DCT Temporal Fusion

Single-frame texture can be insufficient against high-quality video replay
attacks. To capture dynamic information, the system stacks LBP vectors across
multiple frames:

```text
frame_1 -> LBP vector
frame_2 -> LBP vector
...
frame_n -> LBP vector
```

A one-dimensional Discrete Cosine Transform is then applied over the temporal
sequence. The DCT converts frame-to-frame changes into frequency-domain
coefficients, preserving useful temporal patterns while keeping the feature
vector compact.

The prototype in `dct.py` extracts uniform LBP histograms and keeps the first
`20` DCT components per feature column.

### SVM Classification

The target liveness classifier is a Support Vector Machine. It consumes the
LBP-DCT feature matrix and outputs:

```text
real face
spoof / video attack
```

SVM inference is appropriate for resource-constrained devices because it is
cheap compared with large temporal neural models and works well with compact,
handcrafted feature vectors.

`mobile_auth_pipeline.py` supports this path through `--svm-model`, which loads a
pickled model exposing a scikit-learn-style `predict()` method. Because this
repository does not include a liveness dataset or trained SVM weights, the
default demo mode uses temporal DCT energy as a lightweight stand-in threshold.

## Face Recognition

### Regional SIFT Feature Extraction

Scale-Invariant Feature Transform is robust to scale, rotation, and moderate
brightness changes. Full-face SIFT matching can be expensive on mobile devices,
so the target system limits extraction to six facial regions of interest:

- Left eye
- Right eye
- Nose
- Mouth
- Left ear
- Right ear

This reduces descriptor count and matching cost while focusing computation on
discriminative identity regions.

### Weighted Template Matching

Each region is compared with the corresponding enrolled template region. The
system computes inverse distance as a similarity measure:

```text
similarity_i = 1 / (distance(query_i, template_i) + epsilon)
```

The total identity score is a weighted sum:

```text
score = sum(weight_i * similarity_i)
```

The identity branch passes only when the score exceeds the configured threshold.
Weights allow the system to give more influence to stable or discriminative
regions and less influence to noisy or partially occluded regions.

## Parallel Scheduling

The liveness and recognition branches are independent after the face crop is
created. A production implementation can use a thread pool or asynchronous task
runtime:

```python
liveness_future = executor.submit(liveness_engine.predict, face_crop_sequence)
identity_future = executor.submit(recognition_engine.match, face_crop)

liveness_pass = liveness_future.result()
identity_match = identity_future.result()

authenticated = liveness_pass and identity_match
```

The implemented prototype uses this same structure with `ThreadPoolExecutor` in
`MobileAuthenticator.authenticate()`.

The speedup is bounded by the slower branch, shared preprocessing time, and
hardware contention. On mobile devices, profiling should verify that parallel
execution improves latency without unacceptable battery or thermal cost.
