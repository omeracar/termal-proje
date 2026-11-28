#!/usr/bin/env python3
"""Tek kayıtta dört ROI yapılandırmasını karşılaştırır ve basamak kaynağını teşhis eder.

  A) sabit ROI            B) sabit ROI − arka plan
  C) baş takipli ROI      D) baş takipli ROI − arka plan

Arka plan: hiçbir karede sıcak bileşene ait olmayan piksellerin ortalaması (kameranın tüm görüntüye
etki eden kaymalarını yakalar). Baş takibi: en büyük sıcak bileşenin ağırlık merkezi (sezgisel).

DİKKAT: Dört yapılandırmayı deneyip en iyisini seçmek hafif bir seçim yanlılığıdır; tüm satırlar
raporlanır. Karar, birkaç kişide tutarlı sonuca göre verilmeli.

  python scripts/breathing_compare.py --npz kayit.npz --reference kisi_respiration.csv --out-dir sonuc/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.breathing_validation import (  # noqa: E402
    auto_roi_from_mean, celsius_from_raw, count_steps, head_shifts, hot_track, parse_event_times,
    static_background_series, tracked_roi_series, validate_breathing,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--roi", default="auto")
    ap.add_argument("--window", type=float, default=40.0)
    ap.add_argument("--step", type=float, default=20.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz)
    raw, times = data["frames"], data["bag_time_s"]
    n = len(raw)
    fps = 1.0 / float(np.median(np.diff(times)))
    mean_raw = np.zeros(raw.shape[1:], dtype=np.float64)
    for i in range(0, n, 200):
        mean_raw += raw[i:i + 200].sum(axis=0, dtype=np.float64)
    roi = auto_roi_from_mean(celsius_from_raw(mean_raw / n)) if args.roi == "auto" else tuple(int(v) for v in args.roi.split(","))
    x, y, w, h = roi
    events = parse_event_times(Path(args.reference))

    print(f"Kare: {n}, {fps:.1f} Hz, ROI (x,y,w,h): {roi}; baş/arka plan izleniyor…")
    centers, ever_hot = hot_track(raw)
    shifts = head_shifts(centers)
    fixed = celsius_from_raw(raw[:, y:y + h, x:x + w].reshape(n, -1).mean(axis=1, dtype=np.float64))
    tracked = tracked_roi_series(raw, roi, shifts)
    background = static_background_series(raw, ever_hot)
    bg_rel = background - background.mean()
    series = {"A sabit ROI": fixed, "B sabit ROI - arka plan": fixed - bg_rel,
              "C baş takipli ROI": tracked, "D baş takipli ROI - arka plan": tracked - bg_rel}

    dup = float(np.mean(np.diff(fixed) == 0))
    print(f"\nTanı: ardışık özdeş ROI değeri oranı {dup:.1%} (yüksekse kare tekrarı/donma olabilir)")
    print(f"Baş kayması: x aralığı {np.ptp(shifts[:, 0]):.0f} px, y aralığı {np.ptp(shifts[:, 1]):.0f} px; "
          f"merkezi bulunamayan kare: {int(np.isnan(centers[:, 0]).sum())}")
    print("Sıçrama sayısı (>0,4 °C / 0,5 s):", {k: count_steps(v, fps) for k, v in series.items()},
          "| arka plan:", count_steps(background, fps))

    results = {}
    print(f"\n{'yapılandırma':32s} {'alg.oranı':>9s} {'ort.hata/dk':>12s} {'±3 içinde':>10s}")
    for name, values in series.items():
        r = validate_breathing(times, values, events, args.window, args.step)
        results[name] = r["summary"]
        s = r["summary"]
        f = lambda v: "—" if v is None else str(v)
        print(f"{name:32s} {f(s['detection_rate']):>9s} {f(s['mean_abs_error_per_min']):>12s} {f(s['within_tolerance_fraction']):>10s}")
    (out_dir / "compare.json").write_text(json.dumps(
        {"roi": list(roi), "fps": round(fps, 2), "steps": {k: count_steps(v, fps) for k, v in series.items()},
         "background_steps": count_steps(background, fps), "duplicate_fraction": round(dup, 4),
         "head_shift_range_px": [float(np.ptp(shifts[:, 0])), float(np.ptp(shifts[:, 1]))], "results": results},
        indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = times - times[0]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    panels = [(fixed, "A: sabit ROI (°C)"), (background, "Arka plan ortalaması (°C) — tüm görüntüyü etkileyen kayma?"),
              (None, "Baş kayması (piksel): mavi x, turuncu y"), (series["D baş takipli ROI - arka plan"], "D: baş takipli ROI − arka plan (°C)")]
    for ax, (vals, title) in zip(axes, panels):
        if vals is None:
            ax.plot(t, shifts[:, 0], lw=0.8)
            ax.plot(t, shifts[:, 1], lw=0.8)
        else:
            ax.plot(t, vals, lw=0.8)
        for e in events:
            ax.axvline(e - times[0], color="tab:red", alpha=0.2, lw=0.7)
        ax.set_title(title, fontsize=9)
    axes[-1].set_xlabel("Zaman (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostic.png", dpi=120)
    print(f"\nYazıldı: {out_dir / 'compare.json'} ve {out_dir / 'diagnostic.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
