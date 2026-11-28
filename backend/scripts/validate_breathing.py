#!/usr/bin/env python3
"""LCAS termal kaydında solunum algılayıcısını referans solunum kaydıyla karşılaştırır.

Önce rosbag'i dönüştürün:  python tools/lcas_rosbag_to_npz.py KAYIT.bag --out kayit.npz
Sonra:
  python scripts/validate_breathing.py --npz kayit.npz --reference KISI_respiration.csv \
      --roi auto --out ../data/validation/breathing_result.json --plot ../data/validation/breathing.png

--roi: "auto" (en sıcak bölge) ya da "x,y,w,h" (piksel). Gerçek kayıtta ROI'yi gözle kontrol etmek için
--roi-preview ile ortalama kare üzerinde kutuyu çizen bir PNG üretin.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.breathing_validation import (  # noqa: E402
    auto_roi_from_mean, celsius_from_raw, parse_event_times, reference_time_range, validate_breathing,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--reference", required=True, help="Solunum olay CSV'si")
    ap.add_argument("--roi", default="auto")
    ap.add_argument("--time-source", choices=["bag", "header"], default="bag",
                    help="Kare zamanı: rosbag kayıt zamanı ya da mesaj başlığı")
    ap.add_argument("--window", type=float, default=40.0)
    ap.add_argument("--step", type=float, default=20.0)
    ap.add_argument("--offset", type=float, default=0.0, help="Referans saatine eklenecek saniye")
    ap.add_argument("--out")
    ap.add_argument("--plot")
    ap.add_argument("--roi-preview")
    args = ap.parse_args()

    data = np.load(args.npz)
    raw = data["frames"]  # uint16; tamamı float'a çevrilmez (bellek)
    n_frames = len(raw)
    mean_raw = np.zeros(raw.shape[1:], dtype=np.float64)
    for i in range(0, n_frames, 200):
        mean_raw += raw[i:i + 200].sum(axis=0, dtype=np.float64)
    mean_celsius = celsius_from_raw(mean_raw / n_frames)
    times = data["bag_time_s" if args.time_source == "bag" else "header_time_s"]
    roi = auto_roi_from_mean(mean_celsius) if args.roi == "auto" else tuple(int(v) for v in args.roi.split(","))
    events = parse_event_times(Path(args.reference))
    print(f"Kare: {n_frames}, ROI (x,y,w,h): {roi}, referans nefes olayı: {len(events)}")
    print(f"Kayıt aralığı (epoch): {times[0]:.1f} – {times[-1]:.1f}; ilk/son olay: {events[0]:.1f} – {events[-1]:.1f}")
    if events[-1] < times[0] or events[0] > times[-1]:
        print("UYARI: referans zamanları kayıtla örtüşmüyor. Yanlış kişi/saat dilimi olabilir.")

    if args.roi_preview:
        gray = cv2.normalize(mean_celsius, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        x, y, w, h = roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.imwrite(args.roi_preview, cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

    x, y, w, h = roi
    series = celsius_from_raw(raw[:, y:y + h, x:x + w].reshape(n_frames, -1).mean(axis=1, dtype=np.float64))
    ref_first, ref_last, ref_rows = reference_time_range(args.reference)
    print(f"Referans CSV: {ref_rows} satır (kare sayısı {n_frames}); ilk satır − ilk kare = {ref_first - times[0]:+.2f} s")
    result = validate_breathing(times, series, events, args.window, args.step, args.offset)
    result["roi"] = list(roi)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    for w in result["windows"]:
        print(w)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(times - times[0], series, lw=0.8)
        for e in events + args.offset:
            ax.axvline(e - times[0], color="tab:red", alpha=0.25, lw=0.8)
        ax.set_xlabel("Zaman (s)")
        ax.set_ylabel("ROI ortalaması (°C)")
        ax.set_title("ROI sıcaklık serisi (kırmızı çizgiler: referans nefes olayları)")
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=130)
    return 0


if __name__ == "__main__":
    sys.exit(main())
