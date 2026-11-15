"""
Termal görüntülerde insan tespiti (YOLOX, COCO "person" sınıfı).

- OpenCV'nin DNN modülüyle çalışır: PyTorch / ek bağımlılık gerekmez.
- İlk kullanımda model (~36 MB) otomatik indirilir: backend/weights/yolox_s.onnx
- FLIR ADAS gibi veri setlerindeki 8-bit JPEG ve 16-bit TIFF görüntüleri desteklenir.
"""

import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
)
MODEL_PATH = Path(__file__).resolve().parents[2] / "weights" / "yolox_s.onnx"
MODEL_MIN_BYTES = 30 * 1024 * 1024  # bozuk / eksik indirmeyi yakalamak için

INPUT_SIZE = 640
PERSON_CLASS_ID = 0  # COCO


class ModelUnavailableError(RuntimeError):
    """Model dosyası indirilemedi veya yüklenemedi."""


def to_bgr8(image: np.ndarray) -> np.ndarray:
    """Her tür termal görüntüyü (gri, 16-bit, BGRA...) 8-bit 3 kanallı BGR'ye çevirir."""
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        data = image.astype(np.float32)
        lo, hi = np.percentile(data, (1, 99))
        if hi <= lo:
            hi = lo + 1.0
        data = np.clip((data - lo) / (hi - lo), 0.0, 1.0) * 255.0
        image = data.astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _ensure_model(path: Path = MODEL_PATH) -> Path:
    if path.exists() and path.stat().st_size >= MODEL_MIN_BYTES:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if tmp.stat().st_size < MODEL_MIN_BYTES:
            raise ModelUnavailableError("İndirilen model dosyası eksik veya bozuk")
        tmp.replace(path)
    except ModelUnavailableError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # ağ hatası vb.
        tmp.unlink(missing_ok=True)
        raise ModelUnavailableError(
            f"Model indirilemedi ({exc}). Elle indirip şuraya koyun: {path}\n{MODEL_URL}"
        ) from exc
    return path


class PersonDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self._model_path = model_path
        self._net = None
        self._lock = threading.Lock()
        self._grids, self._strides = self._build_grids()

    @staticmethod
    def _build_grids():
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = INPUT_SIZE // stride
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grid = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((grid.shape[0], 1), stride))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def _load(self):
        if self._net is None:
            path = _ensure_model(self._model_path)
            try:
                self._net = cv2.dnn.readNetFromONNX(str(path))
            except cv2.error as exc:
                raise ModelUnavailableError(f"Model yüklenemedi: {exc}") from exc
        return self._net

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.30,
        nms_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Görüntüdeki kişileri döndürür:
        [{"x", "y", "w", "h", "score"}, ...]  (skora göre azalan sırada)
        """
        img = to_bgr8(image)
        h, w = img.shape[:2]

        # Letterbox: oranı koruyarak 640x640'a sığdır
        ratio = min(INPUT_SIZE / h, INPUT_SIZE / w)
        nh, nw = int(round(h * ratio)), int(round(w * ratio))
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(canvas, 1.0, (INPUT_SIZE, INPUT_SIZE), swapRB=False)

        with self._lock:  # cv2.dnn ağı thread-safe değil
            net = self._load()
            net.setInput(blob)
            out = net.forward()

        out = np.squeeze(out, 0).astype(np.float32)  # (8400, 85)
        out[:, :2] = (out[:, :2] + self._grids) * self._strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self._strides

        scores = out[:, 4] * out[:, 5 + PERSON_CLASS_ID]
        keep = scores >= conf_threshold
        if not np.any(keep):
            return []

        cxcywh, scores = out[keep, :4], scores[keep]
        boxes = []
        for cx, cy, bw, bh in cxcywh:
            boxes.append([
                float((cx - bw / 2) / ratio),
                float((cy - bh / 2) / ratio),
                float(bw / ratio),
                float(bh / ratio),
            ])

        idx = cv2.dnn.NMSBoxes(boxes, scores.tolist(), conf_threshold, nms_threshold)
        people = []
        for i in np.array(idx).reshape(-1):
            x, y, bw, bh = boxes[i]
            x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
            x2, y2 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
            if x2 <= x1 or y2 <= y1:
                continue
            people.append({
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "score": round(float(scores[i]), 3),
            })

        people.sort(key=lambda p: p["score"], reverse=True)
        return people
