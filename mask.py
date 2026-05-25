import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# Edit these values directly instead of passing command-line arguments.
SOURCE = 0
SCORE_THRESHOLD = 0.6
MASK_THRESHOLD = 0.5
MAX_DETECTIONS = 10
RESIZE_WIDTH = 960
FORCE_CPU = False


@dataclass
class Detection:
    box: np.ndarray
    label: str
    score: float
    mask: np.ndarray
    color: Tuple[int, int, int]


def make_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(42)
    colors = rng.integers(64, 256, size=(num_classes, 3), dtype=np.uint8)
    return [tuple(int(channel) for channel in color) for color in colors]


def build_model(device: torch.device) -> Tuple[torch.nn.Module, object]:
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    model.eval().to(device)
    return model, weights


def open_source(source: str | int) -> cv2.VideoCapture:
    stream: str | int
    if isinstance(source, str) and source.isdigit():
        stream = int(source)
    else:
        stream = source
    capture = cv2.VideoCapture(stream)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")
    return capture


def resize_frame(frame: np.ndarray, resize_width: int) -> np.ndarray:
    if resize_width <= 0 or frame.shape[1] <= resize_width:
        return frame
    scale = resize_width / frame.shape[1]
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_LINEAR)


def preprocess_frame(
    frame_bgr: np.ndarray,
    device: torch.device,
    use_half: bool,
) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0)
    tensor = tensor.to(device, non_blocking=True)
    if use_half:
        tensor = tensor.half()
    return tensor


def extract_detections(
    prediction: dict,
    score_threshold: float,
    mask_threshold: float,
    max_detections: int,
    palette: List[Tuple[int, int, int]],
) -> List[Detection]:
    scores = prediction["scores"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy().astype(np.int32)
    labels = prediction["labels"].detach().cpu().numpy().astype(np.int32)
    masks = prediction["masks"].detach().cpu().numpy()

    detections: List[Detection] = []
    for score, box, label_idx, mask in zip(scores, boxes, labels, masks):
        if score < score_threshold:
            break
        color = palette[label_idx % len(palette)]
        detections.append(
            Detection(
                box=box,
                label=COCO_INSTANCE_CATEGORY_NAMES[label_idx],
                score=float(score),
                mask=(mask[0] >= mask_threshold),
                color=color,
            )
        )
        if len(detections) >= max_detections:
            break
    return detections


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    overlay = frame.copy()

    for detection in detections:
        overlay[detection.mask] = (
            overlay[detection.mask] * 0.45 + np.array(detection.color) * 0.55
        ).astype(np.uint8)

    output = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    for detection in detections:
        x1, y1, x2, y2 = detection.box.tolist()
        cv2.rectangle(output, (x1, y1), (x2, y2), detection.color, 2)

        text = f"{detection.label}: {detection.score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        y_text = max(y1, text_height + baseline + 6)
        cv2.rectangle(
            output,
            (x1, y_text - text_height - baseline - 6),
            (x1 + text_width + 8, y_text + 4),
            detection.color,
            -1,
        )
        cv2.putText(
            output,
            text,
            (x1 + 4, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def put_status(frame: np.ndarray, fps: float, device: torch.device) -> None:
    text = f"FPS: {fps:.1f} | Device: {device.type.upper()} | Press q to quit"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 240, 20),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    device = torch.device(
        "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda"
    )
    use_half = device.type == "cuda"

    model, _ = build_model(device)
    if use_half:
        model.half()

    capture = open_source(SOURCE)
    palette = make_palette(len(COCO_INSTANCE_CATEGORY_NAMES))

    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = resize_frame(frame, RESIZE_WIDTH)
            tensor = preprocess_frame(frame, device, use_half)

            with torch.inference_mode():
                prediction = model([tensor])[0]

            detections = extract_detections(
                prediction=prediction,
                score_threshold=SCORE_THRESHOLD,
                mask_threshold=MASK_THRESHOLD,
                max_detections=MAX_DETECTIONS,
                palette=palette,
            )

            rendered = draw_detections(frame, detections)

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time
            put_status(rendered, fps, device)

            cv2.imshow("Mask R-CNN Real-Time Segmentation", rendered)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
