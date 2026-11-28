#!/usr/bin/env python3
"""Tanı aracı: nefes bandındaki periyodik sinyal görüntünün HANGİ bölgesinde güçlü?

Görüntüyü KARE x KARE pikselik karelere böler, her karenin ortalama sıcaklık serisine solunum
algılayıcısını uygular ve tepe/arka plan oranını (dB) harita olarak çizer.

UYARI: Yüzlerce kareden en iyisini seçmek çoklu karşılaştırma yanlılığı yaratır. Bu harita yalnızca
sinyalin nerede olabileceğini KEŞFETMEK içindir. Burada bulunan bölgeyle aynı kayıtta ölçüm yapıp
başarı iddia etmek döngüseldir; ROI kuralı önceden belirlenip BAŞKA kayıtlarda sınanmalıdır.

  python scripts/breathing_map.py --npz ../data/validation/claudio.npz \
      --reference /yol/Claudio_respiration.csv --out ../data/validation/breathing_map.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.breathing_validation import (  # noqa: E402
    celsius_from_raw, parse_event_times, reference_rate_per_min, resample_uniform,
)
from app.services.vital_signs import VitalSignsService  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = np.load(args.npz)
    raw, times = data["frames"], data["bag_time_s"]
    n, height, width = raw.shape
    T = args.tile
    th, tw = height // T, width // T
    tiles = np.zeros((n, th, tw), dtype=np.float64)
    for i in range(0, n, 200):
        chunk = raw[i:i + 200, :th * T, :tw * T].astype(np.float64)
        tiles[i:i + 200] = chunk.reshape(len(chunk), th, T, tw, T).mean(axis=(2, 4))
    tiles = celsius_from_raw(tiles)
    flat = tiles.reshape(n, -1)
    grid, _, fps = resample_uniform(times, flat[:, 0])
    service = VitalSignsService()
    score = np.full(th * tw, np.nan)
    rate = np.full(th * tw, np.nan)
    order = np.argsort(times)
    for k in range(th * tw):
        series = np.interp(grid, times[order], flat[order, k])
        details = service.analyze_temperature_series(series.tolist(), fps=fps).get("details") or {}
        if "peak_to_trend_db" in details:
            score[k] = details["peak_to_trend_db"]
            rate[k] = details["respiratory_band_rate_per_min"]

    ref_rate = None
    if args.reference:
        events = parse_event_times(Path(args.reference))
        ref_rate = reference_rate_per_min(events, times[0], times[-1] + 1)
    print(f"Kayıt: {n} kare, {fps:.1f} Hz (ortanca), kare boyu {T} px, ızgara {th}x{tw}")
    if ref_rate:
        print(f"Referans (tüm kayıt, ortanca aralık): {ref_rate:.1f} nefes/dk")
    top = np.argsort(np.nan_to_num(score, nan=-1e9))[::-1][:10]
    print("En yüksek 10 kare (x, y piksel; tepe/arka plan dB; tahmin nefes/dk):")
    for k in top:
        r, c = divmod(int(k), tw)
        tag = ""
        if ref_rate:
            tag = "  <- referansa yakın (±3)" if abs(rate[k] - ref_rate) <= 3 else ""
        print(f"  ({c * T:3d},{r * T:3d})  {score[k]:6.1f} dB  {rate[k]:5.1f}/dk{tag}")
    above = int(np.sum(score >= 10))
    print(f"10 dB üzeri kare sayısı: {above} / {th * tw}  (gürültüde de birkaç kare şans eseri geçebilir)")

    mean_gray = cv2.normalize(raw.mean(axis=0, dtype=np.float64), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    base = cv2.cvtColor(mean_gray, cv2.COLOR_GRAY2BGR)
    s = np.clip(np.nan_to_num(score.reshape(th, tw), nan=-5.0), -5, 25)
    heat = cv2.applyColorMap(((s + 5) / 30 * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.resize(heat, (tw * T, th * T), interpolation=cv2.INTER_NEAREST)
    overlay = base.copy()
    overlay[:th * T, :tw * T] = cv2.addWeighted(base[:th * T, :tw * T], 0.35, heat, 0.65, 0)
    out = np.hstack([base, overlay])
    cv2.imwrite(args.out, cv2.resize(out, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    print(f"Yazıldı: {args.out} (sol: ortalama kare, sağ: periyodiklik haritası; açık sarı = güçlü)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
