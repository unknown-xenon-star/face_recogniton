import argparse
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern


IMG_SIZE = (64, 64)
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
LBP_BINS = np.arange(0, 60)
DCT_COMPONENTS = 20

REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "left_eye": (0.12, 0.18, 0.42, 0.42),
    "right_eye": (0.58, 0.18, 0.88, 0.42),
    "nose": (0.34, 0.34, 0.66, 0.68),
    "mouth": (0.28, 0.64, 0.72, 0.88),
    "left_ear": (0.00, 0.28, 0.20, 0.72),
    "right_ear": (0.80, 0.28, 1.00, 0.72),
}

REGION_WEIGHTS = {
    "left_eye": 0.18,
    "right_eye": 0.18,
    "nose": 0.22,
    "mouth": 0.20,
    "left_ear": 0.11,
    "right_ear": 0.11,
}


@dataclass
class BrightnessState:
    value: float
    threshold: float
    needs_light: bool
    supplemental_gain: float


@dataclass
class DetectionResult:
    frame: np.ndarray
    face_bgr: np.ndarray
    face_gray_64: np.ndarray
    box: Tuple[int, int, int, int]
    brightness: BrightnessState


@dataclass
class LivenessResult:
    passed: bool
    score: float
    feature_shape: Tuple[int, ...]


@dataclass
class RecognitionResult:
    passed: bool
    score: float
    region_scores: Dict[str, float]


@dataclass
class AuthenticationResult:
    authenticated: bool
    liveness: LivenessResult
    recognition: RecognitionResult
    elapsed_ms: float


class Preprocessor:
    def __init__(self, brightness_threshold: float = 80.0) -> None:
        self.brightness_threshold = brightness_threshold

    def estimate_brightness(self, gray: np.ndarray) -> BrightnessState:
        value = float(gray.mean())
        needs_light = value < self.brightness_threshold
        supplemental_gain = max(self.brightness_threshold - value, 0.0)
        return BrightnessState(
            value=value,
            threshold=self.brightness_threshold,
            needs_light=needs_light,
            supplemental_gain=supplemental_gain,
        )

    def preprocess_gray(self, gray: np.ndarray) -> np.ndarray:
        filtered = cv2.GaussianBlur(gray, (5, 5), 1.0)
        equalized = cv2.equalizeHist(filtered)
        return self.grayscale_stretch(equalized)

    @staticmethod
    def grayscale_stretch(gray: np.ndarray) -> np.ndarray:
        min_value = float(np.min(gray))
        max_value = float(np.max(gray))
        if max_value <= min_value:
            return gray.copy()
        stretched = ((gray.astype(np.float32) - min_value) / (max_value - min_value)) * 255.0
        return stretched.astype(np.uint8)


class FaceDetector:
    def __init__(self) -> None:
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.detector.empty():
            raise RuntimeError("Unable to load OpenCV Haar cascade face detector")

    def detect_largest(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        return int(x), int(y), int(w), int(h)


class LivenessEngine:
    def __init__(self, threshold: float = 0.015, svm_model_path: Optional[str] = None) -> None:
        self.threshold = threshold
        self.svm_model = None
        if svm_model_path:
            with open(svm_model_path, "rb") as model_file:
                self.svm_model = pickle.load(model_file)

    def extract_lbp_histogram(self, gray_64: np.ndarray) -> np.ndarray:
        lbp = local_binary_pattern(gray_64, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
        hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, 59))
        hist = hist.astype(np.float32)
        return hist / (hist.sum() + 1e-6)

    def extract_dct_features(self, face_sequence: Sequence[np.ndarray]) -> np.ndarray:
        lbp_matrix = np.array(
            [self.extract_lbp_histogram(face) for face in face_sequence],
            dtype=np.float32,
        )
        features: List[float] = []
        for col in range(lbp_matrix.shape[1]):
            coeff = dct(lbp_matrix[:, col], norm="ortho")[:DCT_COMPONENTS]
            features.extend(float(value) for value in coeff)
        return np.array(features, dtype=np.float32)

    def predict(self, face_sequence: Sequence[np.ndarray]) -> LivenessResult:
        features = self.extract_dct_features(face_sequence)
        if self.svm_model is not None:
            sample = features.reshape(1, -1)
            prediction = int(self.svm_model.predict(sample)[0])
            if hasattr(self.svm_model, "decision_function"):
                score = float(self.svm_model.decision_function(sample).ravel()[0])
            elif hasattr(self.svm_model, "predict_proba"):
                score = float(self.svm_model.predict_proba(sample).ravel()[-1])
            else:
                score = float(prediction)
            return LivenessResult(
                passed=prediction == 1,
                score=score,
                feature_shape=features.shape,
            )

        temporal_energy = float(np.mean(np.abs(features[1:]))) if features.size > 1 else 0.0
        return LivenessResult(
            passed=temporal_energy >= self.threshold,
            score=temporal_energy,
            feature_shape=features.shape,
        )


class RecognitionEngine:
    def __init__(self, match_threshold: float = 0.16) -> None:
        self.match_threshold = match_threshold
        self.extractor, self.norm = self._build_extractor()
        self.matcher = cv2.BFMatcher(self.norm, crossCheck=True)

    @staticmethod
    def _build_extractor():
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(), cv2.NORM_L2
        return cv2.ORB_create(nfeatures=500), cv2.NORM_HAMMING

    def extract_template(self, face_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        template: Dict[str, np.ndarray] = {}
        for name, crop in crop_regions(gray).items():
            _, descriptors = self.extractor.detectAndCompute(crop, None)
            if descriptors is None:
                dtype = np.uint8 if self.norm == cv2.NORM_HAMMING else np.float32
                descriptors = np.empty((0, self.extractor.descriptorSize()), dtype=dtype)
            template[name] = descriptors
        return template

    def match(self, face_bgr: np.ndarray, template: Dict[str, np.ndarray]) -> RecognitionResult:
        query = self.extract_template(face_bgr)
        region_scores: Dict[str, float] = {}

        for name, weight in REGION_WEIGHTS.items():
            query_desc = query.get(name)
            template_desc = template.get(name)
            if query_desc is None or template_desc is None or len(query_desc) == 0 or len(template_desc) == 0:
                region_scores[name] = 0.0
                continue

            matches = self.matcher.match(query_desc, template_desc)
            if not matches:
                region_scores[name] = 0.0
                continue

            distances = np.array([match.distance for match in matches], dtype=np.float32)
            best = np.sort(distances)[: min(12, len(distances))]
            region_scores[name] = float(weight / (1.0 + best.mean()))

        score = float(sum(region_scores.values()))
        return RecognitionResult(
            passed=score >= self.match_threshold,
            score=score,
            region_scores=region_scores,
        )

    def save_template(self, path: Path, template: Dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "extractor": self.extractor.__class__.__name__,
            "regions": list(REGIONS.keys()),
            "weights": REGION_WEIGHTS,
        }
        arrays = {name: descriptors for name, descriptors in template.items()}
        arrays["metadata"] = np.array(json.dumps(metadata))
        np.savez_compressed(path, **arrays)

    @staticmethod
    def load_template(path: Path) -> Dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        data = np.load(path, allow_pickle=False)
        return {name: data[name] for name in REGIONS.keys() if name in data}


class MobileAuthenticator:
    def __init__(
        self,
        brightness_threshold: float = 80.0,
        liveness_threshold: float = 0.015,
        recognition_threshold: float = 0.16,
        svm_model_path: Optional[str] = None,
    ) -> None:
        self.preprocessor = Preprocessor(brightness_threshold)
        self.detector = FaceDetector()
        self.liveness = LivenessEngine(liveness_threshold, svm_model_path)
        self.recognition = RecognitionEngine(recognition_threshold)

    def process_frame(self, frame: np.ndarray) -> Optional[DetectionResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = self.preprocessor.estimate_brightness(gray)
        processed_gray = self.preprocessor.preprocess_gray(gray)
        box = self.detector.detect_largest(processed_gray)
        if box is None:
            return None

        x, y, w, h = box
        face_bgr = frame[y : y + h, x : x + w]
        if face_bgr.size == 0:
            return None

        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        face_gray_64 = cv2.resize(face_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
        return DetectionResult(
            frame=frame,
            face_bgr=face_bgr,
            face_gray_64=face_gray_64,
            box=box,
            brightness=brightness,
        )

    def authenticate(
        self,
        face_sequence: Sequence[np.ndarray],
        latest_face_bgr: np.ndarray,
        template: Dict[str, np.ndarray],
    ) -> AuthenticationResult:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            liveness_future = executor.submit(self.liveness.predict, face_sequence)
            recognition_future = executor.submit(self.recognition.match, latest_face_bgr, template)
            liveness = liveness_future.result()
            recognition = recognition_future.result()

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return AuthenticationResult(
            authenticated=liveness.passed and recognition.passed,
            liveness=liveness,
            recognition=recognition,
            elapsed_ms=elapsed_ms,
        )


def crop_regions(gray: np.ndarray) -> Dict[str, np.ndarray]:
    height, width = gray.shape[:2]
    regions: Dict[str, np.ndarray] = {}
    for name, (x1, y1, x2, y2) in REGIONS.items():
        left = max(int(x1 * width), 0)
        top = max(int(y1 * height), 0)
        right = min(int(x2 * width), width)
        bottom = min(int(y2 * height), height)
        regions[name] = gray[top:bottom, left:right]
    return regions


def open_capture(source: str) -> cv2.VideoCapture:
    capture_source: int | str = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")
    return capture


def collect_detections(
    authenticator: MobileAuthenticator,
    source: str,
    frames: int,
    preview: bool,
) -> List[DetectionResult]:
    capture = open_capture(source)
    detections: List[DetectionResult] = []

    try:
        while len(detections) < frames:
            ok, frame = capture.read()
            if not ok:
                break

            detection = authenticator.process_frame(frame)
            display = frame.copy()

            if detection is not None:
                detections.append(detection)
                x, y, w, h = detection.box
                color = (0, 255, 0)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                light_text = "light needed" if detection.brightness.needs_light else "light ok"
                cv2.putText(
                    display,
                    f"{len(detections)}/{frames} | {light_text} | B={detection.brightness.value:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            if preview:
                cv2.imshow("Mobile Authentication Pipeline", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if preview:
            cv2.destroyAllWindows()

    return detections


def enroll(args: argparse.Namespace) -> None:
    authenticator = MobileAuthenticator(
        brightness_threshold=args.brightness_threshold,
        liveness_threshold=args.liveness_threshold,
        recognition_threshold=args.recognition_threshold,
        svm_model_path=args.svm_model,
    )
    detections = collect_detections(authenticator, args.source, 1, args.preview)
    if not detections:
        raise RuntimeError("No face detected for enrollment")

    template = authenticator.recognition.extract_template(detections[-1].face_bgr)
    authenticator.recognition.save_template(Path(args.template), template)
    print(f"Saved identity template: {args.template}")


def authenticate(args: argparse.Namespace) -> None:
    authenticator = MobileAuthenticator(
        brightness_threshold=args.brightness_threshold,
        liveness_threshold=args.liveness_threshold,
        recognition_threshold=args.recognition_threshold,
        svm_model_path=args.svm_model,
    )
    template = RecognitionEngine.load_template(Path(args.template))
    detections = collect_detections(authenticator, args.source, args.frames, args.preview)
    if len(detections) < 2:
        raise RuntimeError("At least two detected face frames are required for liveness")

    face_sequence = [detection.face_gray_64 for detection in detections]
    latest_face = detections[-1].face_bgr
    result = authenticator.authenticate(face_sequence, latest_face, template)

    print(f"Authenticated: {result.authenticated}")
    print(f"Parallel branch time: {result.elapsed_ms:.2f} ms")
    print(f"Liveness: pass={result.liveness.passed} score={result.liveness.score:.6f}")
    print(f"Recognition: pass={result.recognition.passed} score={result.recognition.score:.6f}")
    print("Region scores:")
    for name, score in result.recognition.region_scores.items():
        print(f"  {name}: {score:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel mobile face authentication prototype"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--source", default="0", help="Camera index or video path")
        subparser.add_argument("--template", default="templates/default_identity.npz")
        subparser.add_argument("--brightness-threshold", type=float, default=80.0)
        subparser.add_argument("--liveness-threshold", type=float, default=0.015)
        subparser.add_argument("--recognition-threshold", type=float, default=0.16)
        subparser.add_argument("--svm-model", help="Optional pickled SVM liveness model")
        subparser.add_argument("--no-preview", action="store_false", dest="preview")
        subparser.set_defaults(preview=True)

    enroll_parser = subparsers.add_parser("enroll", help="Create an identity template")
    add_common(enroll_parser)
    enroll_parser.set_defaults(func=enroll)

    auth_parser = subparsers.add_parser("auth", help="Run parallel authentication")
    add_common(auth_parser)
    auth_parser.add_argument("--frames", type=int, default=24)
    auth_parser.set_defaults(func=authenticate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
