"""Video işleme hattı: kare örnekleme, sezgisel ilgi bölgesi (ROI) eşleştirme ve zamansal termal sinyal çıkarımı.

İzler aday bölgelerdir, doğrulanmış kişi değildir. Kameraya özgü bir radyometrik sıcaklık akışı
verilmedikçe (burada uygulanmamıştır) piksel yoğunluğu DN cinsindendir.
"""
from typing import Any, Dict, List
import cv2
import numpy as np
from scipy.interpolate import interp1d

from app.services.image_processing import ImageProcessingService, to_bgr8, to_gray
from app.services.vital_signs import VitalSignsService, with_finding
import base64


class ThermalVideoAnalysisService:
    def __init__(self):
        self.detector = ImageProcessingService()
        self.vital = VitalSignsService()

    def analyze(self, video_path: str, max_sample_rate_hz: float = 10.0) -> Dict[str, Any]:
        if not np.isfinite(max_sample_rate_hz) or max_sample_rate_hz <= 0:
            raise ValueError("max_sample_rate_hz sıfırdan büyük olmalıdır")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Video açılamadı. Codec ve dosya bütünlüğünü kontrol edin.")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
            cap.release()
            raise ValueError("Video üst verisinde geçerli FPS/kare sayısı yok.")

        duration = frame_count / fps
        if duration > 600:
            cap.release()
            raise ValueError("Video süresi 10 dakikalık analiz sınırını aşıyor")
        if width <= 0 or height <= 0 or width * height > 3840 * 2160:
            cap.release()
            raise ValueError("Video boyutları geçersiz ya da 4K sınırını aşıyor")
        stride = max(1, int(round(fps / max_sample_rate_hz)))
        sampled_fps = fps / stride
        tracks: Dict[int, Dict[str, Any]] = {}
        next_id = 1
        frame_index = 0
        sampled_frames = 0
        diagonal = float(np.hypot(width, height))
        max_match_distance = max(20.0, diagonal * 0.12)
        # Önizleme için en çok 40 küçültülmüş kare saklanır (bellek sınırı)
        estimated_samples = max(1, frame_count // stride)
        keep_every = max(1, int(np.ceil(estimated_samples / 40)))
        preview_scale = min(1.0, 480.0 / max(width, 1))
        stored: Dict[int, Dict[str, Any]] = {}

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    sampled_frames += 1
                    candidates, _ = self.detector.detect_people(frame)
                    gray = to_gray(frame)
                    assignments = set()
                    sample_boxes: List[Any] = []

                    # Mevcut izlere açgözlü en yakın merkez eşleştirmesi.
                    for candidate in candidates:
                        cx = candidate["x"] + candidate["w"] / 2.0
                        cy = candidate["y"] + candidate["h"] / 2.0
                        best_id, best_distance = None, max_match_distance
                        for track_id, track in tracks.items():
                            if track_id in assignments or sampled_frames - track["last_sample"] > 3:
                                continue
                            tx, ty = track["last_center"]
                            distance = float(np.hypot(cx - tx, cy - ty))
                            if distance < best_distance:
                                best_id, best_distance = track_id, distance

                        if best_id is None:
                            best_id = next_id
                            next_id += 1
                            tracks[best_id] = {
                                "samples": [], "last_center": (cx, cy),
                                "last_sample": sampled_frames, "detections": 0,
                            }

                        x, y, w, h = (int(candidate[k]) for k in ("x", "y", "w", "h"))
                        roi = gray[y:y + h, x:x + w]
                        if roi.size == 0:
                            continue
                        track = tracks[best_id]
                        track["samples"].append((sampled_frames - 1, float(np.mean(roi))))
                        track["last_center"] = (cx, cy)
                        track["last_sample"] = sampled_frames
                        track["detections"] += 1
                        assignments.add(best_id)
                        sample_boxes.append((best_id, x, y, w, h))
                    if (sampled_frames - 1) % keep_every == 0:
                        small = cv2.resize(frame, None, fx=preview_scale, fy=preview_scale) \
                            if preview_scale < 1.0 else frame.copy()
                        stored[sampled_frames - 1] = {"frame": small, "boxes": sample_boxes}
                frame_index += 1
        finally:
            cap.release()

        results = []
        total_samples = sampled_frames
        for track_id, track in tracks.items():
            observations = track["samples"]
            if len(observations) < 2:
                continue
            indices = np.asarray([v[0] for v in observations], dtype=np.float64)
            values = np.asarray([v[1] for v in observations], dtype=np.float64)
            coverage = len(values) / max(total_samples, 1)
            # Uzun örtülmeleri ve çoğunlukla eksik tespitleri köprüleme.
            if coverage < 0.70:
                vital_result = with_finding({
                    "status": "insufficient_data",
                    "message": "İz kapsama oranı %70'in altında; zamansal analiz yapılmadı.",
                })
            else:
                timeline = np.arange(total_samples, dtype=np.float64)
                series = interp1d(indices, values, kind="linear",
                                  bounds_error=False, fill_value="extrapolate")(timeline)
                vital_result = self.vital.analyze_temperature_series(
                    series.tolist(), fps=sampled_fps
                )
            results.append({
                "track_id": track_id,
                "candidate_observations": len(observations),
                "coverage": round(float(coverage), 3),
                "signal_unit": "DN",
                "signal_description": "İzlenen aday ROI içindeki ortalama gri seviye piksel yoğunluğu",
                "vital_signs": vital_result,
            })

        kept_ids = {r["track_id"] for r in results}
        preview_frames = self._build_previews(stored, kept_ids, preview_scale, sampled_fps)

        return {
            "preview_frames": preview_frames,
            "video": {
                "width": width, "height": height, "source_fps": fps,
                "frame_count": frame_count, "duration_s": round(duration, 2),
                "sampled_frames": sampled_frames, "analysis_fps": round(sampled_fps, 3),
            },
            "tracking_method": "greedy_nearest_centroid_heuristic",
            "track_count": len(results),
            "tracks": results,
            "interpretation_warning": (
                "İzler sezgisel termal aday bölgelerdir, doğrulanmış kişi değildir. "
                "Sinyal piksel yoğunluğudur (DN), kalibre edilmiş sıcaklık değildir. "
                "Periyodik bileşenler nefes almanın veya yaşamın kanıtı değildir; bulunmamaları da ölümün kanıtı değildir."
            ),
        }

    @staticmethod
    def _build_previews(stored, kept_ids, scale, sampled_fps):
        """Saklanan kareler arasından ilk, orta ve son kareyi seçip iz kutularıyla çizer."""
        if not stored:
            return []
        keys = sorted(stored)
        chosen = sorted({keys[0], keys[len(keys) // 2], keys[-1]})
        previews = []
        for key in chosen:
            frame = to_bgr8(stored[key]["frame"]).copy()
            for track_id, x, y, w, h in stored[key]["boxes"]:
                if track_id not in kept_ids:
                    continue
                x0, y0, x1, y1 = (int(round(v * scale)) for v in (x, y, x + w, y + h))
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 0), 2)
                cv2.putText(frame, f"#{track_id}", (x0, max(14, y0 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2, cv2.LINE_AA)
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                previews.append({
                    "time_s": round(key / sampled_fps, 2),
                    "image": "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii"),
                })
        return previews
