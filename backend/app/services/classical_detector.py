"""
Klasik görüntü işleme ile "insan olabilecek bölge" adaylarının tespiti.

Yapay zekâ / makine öğrenmesi / hazır eğitilmiş model KULLANILMAZ; ağdan model indirilmez.
Çıktı, insan olabilecek bölgelerdir; kesin "kişi" sınıflandırması DEĞİLDİR.

Kullanılan teknikler (hepsi klasik):
  1. Gürültü azaltma          : Non-local means + Gauss filtresi
  2. Yerel sıcaklık kontrastı : görüntü - büyük ölçekli arka plan (Gauss)
  3. Eşikleme                 : kontrast dağılımına göre uyarlamalı, çok seviyeli eşik
  4. Morfolojik işlemler      : açma (benek temizleme) + dikey kapama (gövde parçalarını birleştirme)
  5. Bağlı bileşen / kontur   : bölge istatistikleri
  6. Geometrik analiz         : yükseklik, en/boy oranı, doluluk, figür-zemin kontrastı
  7. Yan yana kişileri ayırma : dikey izdüşüm profili
  8. Parça birleştirme        : üst üste duran baş/gövde/bacak parçalarını tek adayda toplama
  9. Arayüz grafiği eleme     : renk skalası, metin ve simge gibi saf siyah+beyaz bölgeler

Bilinen sınırlar (bkz. README): hayvanlar, sıcak nesneler ve yoğun kalabalıklar ayırt edilemez;
parametreler az sayıda görüntüde ayarlanmıştır ve etiketli veriyle doğrulanmalıdır.
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Ayarlar
# --------------------------------------------------------------------------
PARAMS = {
    "denoise_h": 8,            # non-local means gücü
    "bg_sigma_ratio": 0.035,   # arka plan Gauss sigma'sı = bu oran * görüntü yüksekliği (+ min 8)
    "thresh_std_k": 0.9,       # eşik = k * kontrast std'si
    "thresh_min": 12.0,        # eşiğin alt sınırı
    "min_h_ratio": 0.05,       # kişi yüksekliği alt sınırı (görüntü yüksekliği oranı)
    "max_h_ratio": 0.85,       # kişi yüksekliği üst sınırı (yakın plan çekimler için geniş)
    "min_aspect": 1.3,         # yükseklik / genişlik alt sınırı
    "max_aspect": 4.2,
    "min_extent": 0.25,        # alan / kutu alanı alt sınırı
    "max_extent": 0.88,        # tam dikdörtgen (renk skalası, arayüz kutuları) insan değildir
    "min_area": 60,            # piksel
    "split_aspect": 1.8,       # bundan "tombul" kutular birden çok kişi olabilir -> ayırmayı dene
    "n_levels": 8,             # çok seviyeli eşikleme: kaç farklı eşik denenecek
    "level_span": 3.5,         # en yüksek eşik = en düşük eşik * bu çarpan
    "contain_thr": 0.6,        # küçük kutunun bu oranı büyük kutunun içindeyse tekrar sayılır
    "close_ratios": (0.04, 0.12),  # dikey kapama boyları (görüntü yüksekliği oranı)
    "height_ref": 0.17,        # skor: bu yükseklik oranında tam puan (parça/küçük kutuları geri plana atar)
    "w_aspect": 0.25, "w_extent": 0.10, "w_contrast": 0.20, "w_height": 0.20, "w_ring": 0.25,
    "min_score": 0.85,         # bu skorun altındaki adaylar insan sayılmaz
    "ring_ref": 35.0,          # figür-zemin parlaklık farkının tam puan aldığı değer
    "contrast_ref": 45.0,
    "border_strip": 0.06,      # üst/alt şerit: zaman damgası, simge gibi arayüz öğeleri
    "merge_xoverlap": 0.3,     # üst üste duran parçaların yatay örtüşme oranı
    "merge_gap": 0.4,          # parçalar arası dikey boşluk (yükseklik oranı)
    "merge_max_width": 1.6,    # birleşik kutu, en geniş parçanın bu katından geniş olamaz      # skor için: bölge içi ortalama kontrastın tam puan aldığı değer
}


class ClassicalPersonDetector:
    def __init__(self, params: Dict[str, float] = None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)

    # ---------------- 1-3: ön işleme, kontrast, eşik ----------------
    def preprocess(self, gray: np.ndarray) -> np.ndarray:
        p = self.p
        d = cv2.fastNlMeansDenoising(gray, None, h=p["denoise_h"],
                                     templateWindowSize=5, searchWindowSize=15)
        return cv2.GaussianBlur(d, (3, 3), 0)

    def contrast_map(self, denoised: np.ndarray) -> np.ndarray:
        h = denoised.shape[0]
        sigma = max(8.0, self.p["bg_sigma_ratio"] * h)
        f = denoised.astype(np.float32)
        return f - cv2.GaussianBlur(f, (0, 0), sigma)

    def base_threshold(self, contrast: np.ndarray) -> float:
        return max(self.p["thresh_min"], self.p["thresh_std_k"] * float(np.std(contrast)))

    def binarize(self, contrast: np.ndarray, thr: float = None, close_ratio: float = None) -> np.ndarray:
        p = self.p
        if thr is None:
            thr = self.base_threshold(contrast)
        mask = (contrast > thr).astype(np.uint8) * 255

        # 4: morfoloji
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        h = mask.shape[0]
        kh = max(9, int(h * (close_ratio or 0.04))) | 1   # dikey kapama: gövde/bacak parçalarını bağlar
        kw = max(3, int(h * 0.008)) | 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
        return mask

    # ---------------- 7: yan yana kişileri ayırma ----------------
    @staticmethod
    def _split_columns(mask_roi: np.ndarray, min_w: int) -> List[Tuple[int, int]]:
        """Dikey izdüşüm profilindeki çukurlardan keserek (x0, x1) aralıkları döndürür."""
        prof = (mask_roi > 0).sum(axis=0).astype(np.float32)
        prof = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), max(1.0, len(prof) * 0.04)).ravel()
        if prof.max() <= 0:
            return [(0, len(prof))]

        cuts = []
        for x in range(min_w, len(prof) - min_w):
            if prof[x] <= prof[x - 1] and prof[x] < prof[x + 1] and prof[x] < 0.55 * prof.max():
                # iki yanında da belirgin tepe olmalı
                left, right = prof[:x].max(), prof[x + 1:].max()
                if left > 1.6 * prof[x] and right > 1.6 * prof[x]:
                    cuts.append(x)
        bounds = [0] + cuts + [len(prof)]
        segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= min_w]
        return segs or [(0, len(prof))]

    # ---------------- 5-6: bileşenler ve geometrik analiz ----------------
    def detect(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """8-bit tek kanallı (uint8, 2B) görüntüde aday bölgeleri döndürür."""
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("ClassicalPersonDetector.detect() 8-bit tek kanallı görüntü bekler")
        H, W = gray.shape
        den = self.preprocess(gray)
        contrast = self.contrast_map(den)
        t0 = self.base_threshold(contrast)
        self._den = den.astype(np.float32)
        self._gray = gray

        candidates: List[Dict[str, Any]] = []
        for thr in np.linspace(t0, t0 * self.p["level_span"], int(self.p["n_levels"])):
            for cr in self.p["close_ratios"]:
                mask = self.binarize(contrast, float(thr), cr)
                candidates += self._filter_components(mask, contrast, H, W)

        return self._merge_stacked(self._select(candidates), H)

    def _filter_components(self, mask, contrast, H, W) -> List[Dict[str, Any]]:
        p = self.p
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes: List[Dict[str, Any]] = []

        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < p["min_area"]:
                continue
            blob = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255

            # Tombul (geniş) bölge: yan yana kişiler olabilir -> ayır
            parts = [(0, w)]
            if h / max(w, 1) < p["split_aspect"] and w > 0.25 * h:
                parts = self._split_columns(blob, min_w=max(6, int(0.18 * h)))

            for x0, x1 in parts:
                sub = blob[:, x0:x1]
                ys, xs = np.nonzero(sub)
                if len(ys) < p["min_area"]:
                    continue
                by0, by1 = ys.min(), ys.max() + 1
                bx0, bx1 = xs.min(), xs.max() + 1
                bw, bh = bx1 - bx0, by1 - by0
                gx, gy = x + x0 + bx0, y + by0
                box = self._score_box(sub[by0:by1, bx0:bx1], contrast[gy:gy + bh, gx:gx + bw],
                                      gx, gy, bw, bh, H, W)
                if box:
                    boxes.append(box)
        return boxes

    def _select(self, cands):
        """Aynı kişiyi temsil eden iç içe/çakışan adaylardan en iyi skorluyu seç."""
        cands = sorted((b for b in cands if b["score"] >= self.p["min_score"]),
                       key=lambda b: b["score"], reverse=True)
        kept = []
        for b in cands:
            if all(self._iou(b, k) < 0.3 and self._contained(b, k) < self.p["contain_thr"]
                   and self._contained(k, b) < self.p["contain_thr"] for k in kept):
                kept.append(b)
        return kept

    # ---------------- parçalanmış kişileri birleştirme ----------------
    def _merge_stacked(self, boxes, H):
        """Üst üste duran parçaları (baş-gövde / gövde-bacak) tek kişide birleştirir."""
        p = self.p
        boxes = [dict(b) for b in boxes]
        changed = True
        while changed:
            changed = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
                    if ox / float(min(a["w"], b["w"])) < p["merge_xoverlap"]:
                        continue
                    gap = max(a["y"], b["y"]) - min(a["y"] + a["h"], b["y"] + b["h"])
                    if gap > p["merge_gap"] * max(a["h"], b["h"]):
                        continue
                    x0, y0 = min(a["x"], b["x"]), min(a["y"], b["y"])
                    x1 = max(a["x"] + a["w"], b["x"] + b["w"])
                    y1 = max(a["y"] + a["h"], b["y"] + b["h"])
                    w, h = x1 - x0, y1 - y0
                    if w > p["merge_max_width"] * max(a["w"], b["w"]):
                        continue
                    if not (p["min_aspect"] <= h / float(w) <= p["max_aspect"]) or h > p["max_h_ratio"] * H:
                        continue
                    boxes[i] = {"x": int(x0), "y": int(y0), "w": int(w), "h": int(h),
                                "score": max(a["score"], b["score"])}
                    del boxes[j]
                    changed = True
                    break
                if changed:
                    break
        return sorted(boxes, key=lambda b: b["score"], reverse=True)

    @staticmethod
    def _contained(a, b) -> float:
        """a'nın alanının ne kadarı b'nin içinde."""
        iw = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
        ih = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
        if iw <= 0 or ih <= 0:
            return 0.0
        return iw * ih / float(a["w"] * a["h"])

    def _score_box(self, sub, crop_contrast, x, y, w, h, H, W):
        p = self.p
        if h < p["min_h_ratio"] * H or h > p["max_h_ratio"] * H:
            return None
        if y <= 1 or w < 8:          # üst kenara değen şeritler: direk/bina/ağaç
            return None
        strip = p["border_strip"] * H
        if y + h <= strip or y >= H - strip:   # yalnızca üst/alt şeritte: arayüz öğesi
            return None
        box = self._gray[y:y + h, x:x + w]
        if (box <= 3).mean() > 0.15 and (box >= 250).mean() > 0.02:
            return None   # saf siyah + saf beyaz: renk skalası / metin / simge gibi arayüz grafiği
        aspect = h / max(w, 1)
        if not (p["min_aspect"] <= aspect <= p["max_aspect"]):
            return None
        extent = float((sub > 0).sum()) / (w * h)
        if not (p["min_extent"] <= extent <= p["max_extent"]):
            return None

        # Geometri skoru (0-1): en/boy oranı insana yakın mı, doluluk uygun mu
        aspect_score = max(0.0, 1.0 - abs(aspect - 2.6) / 2.6)
        extent_score = max(0.0, 1.0 - abs(extent - 0.55) / 0.55)
        inside = crop_contrast[sub > 0]
        contrast_score = float(min(1.0, max(0.0, inside.mean() / p["contrast_ref"]))) if inside.size else 0.0
        height_score = min(1.0, h / (p["height_ref"] * H))

        # Figür-zemin: kutu içi parlaklık - çevre halkası parlaklığı
        den = self._den
        px, py = max(6, int(0.35 * w)), max(4, int(0.08 * h))
        X0, X1, Y0, Y1 = max(0, x - px), min(W, x + w + px), max(0, y - py), min(H, y + h + py)
        region = den[Y0:Y1, X0:X1]
        ring = np.ones(region.shape, bool)
        ring[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = False
        inner = den[y:y + h, x:x + w][sub > 0]
        ring_score = 0.0
        if inner.size and ring.any():
            ring_score = float(min(1.0, max(0.0, (inner.mean() - region[ring].mean()) / p["ring_ref"])))

        score = round(p["w_aspect"] * aspect_score + p["w_extent"] * extent_score
                      + p["w_contrast"] * contrast_score + p["w_height"] * height_score
                      + p["w_ring"] * ring_score, 3)
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": float(score)}

    @staticmethod
    def _iou(a, b) -> float:
        ax2, ay2, bx2, by2 = a["x"] + a["w"], a["y"] + a["h"], b["x"] + b["w"], b["y"] + b["h"]
        iw = min(ax2, bx2) - max(a["x"], b["x"])
        ih = min(ay2, by2) - max(a["y"], b["y"])
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        return inter / float(a["w"] * a["h"] + b["w"] * b["h"] - inter)
