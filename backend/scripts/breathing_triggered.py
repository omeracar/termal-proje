#!/usr/bin/env python3
"""Olay tetikli ortalama testi: sıcaklık, referans nefes anlarıyla eşleşiyor mu?

Her nefes olayının çevresindeki (varsayılan -1,5 .. +3,5 sn) kare dilimleri, her piksel karesi için
olay başına ortalamaya göre ortalanır ve ortalaması alınır. Nefesle bağlı bir sıcaklık dalgası varsa
ortalama eğride tutarlı bir tepe/çukur oluşur (38 olayla gürültü ~6 kat bastırılır).

İstatistiksel anlamlılık: olay dizisi kayıt içinde rastgele dairesel kaydırılarak (nefesle ilişkisi
bozulur) boş dağılım üretilir. Tüm kareler üzerindeki EN BÜYÜK genlik, kaydırmalardaki en büyük
genliklerle kıyaslanır (çoklu karşılaştırmaya karşı düzeltilmiş p değeri).

Bu bir TANI testidir: referansı kullanır, bu yüzden algılayıcının başarımı DEĞİLDİR. Soru yalnızca
"bu kayıtta nefesle bağlı bir termal sinyal var mı ve nerede?".

  python scripts/breathing_triggered.py --npz ../data/validation/claudio.npz \
      --reference /yol/Claudio_respiration.csv --tile 4 --out ../data/validation/triggered.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.breathing_validation import celsius_from_raw, parse_event_times  # noqa: E402


def tile_means(raw: np.ndarray, tile: int) -> np.ndarray:
    n, height, width = raw.shape
    th, tw = height // tile, width // tile
    out = np.zeros((n, th * tw), dtype=np.float32)
    for i in range(0, n, 200):
        chunk = raw[i:i + 200, :th * tile, :tile * tw].astype(np.float32)
        out[i:i + 200] = chunk.reshape(len(chunk), th, tile, tw, tile).mean(axis=(2, 4)).reshape(len(chunk), -1)
    return celsius_from_raw(out)


def triggered_average(tiles, times, event_times, pre_frames, post_frames):
    """(L, T) ortalama eğri; her olay kendi penceresinin ortalamasına göre ortalanır."""
    idx = np.searchsorted(times, event_times)
    valid = (idx - pre_frames >= 0) & (idx + post_frames < len(times))
    idx = idx[valid]
    if idx.size < 5:
        return None, int(idx.size)
    offsets = np.arange(-pre_frames, post_frames + 1)
    windows = tiles[idx[:, None] + offsets[None, :]]          # (E, L, T)
    windows = windows - windows.mean(axis=1, keepdims=True)    # sürüklenmeyi at
    return windows.mean(axis=0), int(idx.size)


def amplitude(curve):
    return curve.max(axis=0) - curve.min(axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--tile", type=int, default=4)
    ap.add_argument("--pre", type=float, default=1.5)
    ap.add_argument("--post", type=float, default=3.5)
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = np.load(args.npz)
    raw, times = data["frames"], data["bag_time_s"]
    fps = 1.0 / float(np.median(np.diff(times)))
    pre, post = int(args.pre * fps), int(args.post * fps)
    tiles = tile_means(raw, args.tile)
    events = parse_event_times(Path(args.reference))
    height, width = raw.shape[1:]
    th, tw = height // args.tile, width // args.tile

    curve, used = triggered_average(tiles, times, events, pre, post)
    if curve is None:
        print("Yeterli olay yok:", used)
        return 2
    amp = amplitude(curve)
    print(f"Kare: {len(times)}, {fps:.1f} Hz, kare boyu {args.tile} px, olay: {used}/{len(events)} kullanıldı")

    rng = np.random.default_rng(0)
    span = times[-1] - times[0]
    null_max, null_mean, null_std_acc = [], np.zeros_like(amp), np.zeros_like(amp)
    null_amps = []
    for _ in range(args.shuffles):
        shift = rng.uniform(0.1 * span, 0.9 * span)
        shifted = times[0] + np.mod(events - times[0] + shift, span)
        c, _ = triggered_average(tiles, times, np.sort(shifted), pre, post)
        a = amplitude(c)
        null_amps.append(a)
        null_max.append(a.max())
    null_amps = np.asarray(null_amps)
    mu, sd = null_amps.mean(axis=0), null_amps.std(axis=0) + 1e-9
    z = (amp - mu) / sd
    p_fwer = (1 + int(np.sum(np.asarray(null_max) >= amp.max()))) / (1 + args.shuffles)
    print(f"Gözlenen en büyük genlik: {amp.max():.3f} °C | kaydırmalarda en büyük genlik: "
          f"ortalama {np.mean(null_max):.3f}, en yüksek {np.max(null_max):.3f}")
    print(f"Çoklu karşılaştırma düzeltmeli p değeri: {p_fwer:.3f}  (en küçük mümkün: {1 / (1 + args.shuffles):.3f})")
    top = np.argsort(z)[::-1][:10]
    print("En yüksek 10 kare (x, y piksel; genlik °C; z; tepe gecikmesi sn):")
    lags = (np.arange(-pre, post + 1) / fps)
    for k in top:
        r, c = divmod(int(k), tw)
        lag = lags[int(np.argmax(np.abs(curve[:, k])))]
        print(f"  ({c * args.tile:3d},{r * args.tile:3d})  {amp[k]:.3f} °C  z={z[k]:5.1f}  gecikme {lag:+.2f} s")
    print("YORUM: p < 0.05 ve yan yana kümelenmiş parlak kareler varsa nefesle bağlı bir termal sinyal VARDIR.")
    print("       p yüksekse bu kayıtta (bu çözünürlükte) nefesle bağlı sinyal ayırt edilemiyor.")

    mean_gray = cv2.normalize(raw.mean(axis=0, dtype=np.float64), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    base = cv2.cvtColor(mean_gray, cv2.COLOR_GRAY2BGR)
    zi = np.clip(z.reshape(th, tw), -2, 12)
    heat = cv2.applyColorMap(((zi + 2) / 14 * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.resize(heat, (tw * args.tile, th * args.tile), interpolation=cv2.INTER_NEAREST)
    overlay = base.copy()
    overlay[:th * args.tile, :tw * args.tile] = cv2.addWeighted(
        base[:th * args.tile, :tw * args.tile], 0.35, heat, 0.65, 0)
    cv2.imwrite(args.out, cv2.resize(np.hstack([base, overlay]), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for k in top[:5]:
            ax.plot(lags, curve[:, k], lw=1)
        ax.axvline(0, color="k", ls=":")
        ax.set_xlabel("Nefes olayına göre zaman (s)")
        ax.set_ylabel("Ortalamaya göre sıcaklık (°C)")
        ax.set_title("En yüksek 5 karenin olay tetikli ortalama eğrisi")
        fig.tight_layout()
        fig.savefig(str(Path(args.out).with_suffix("")) + "_egriler.png", dpi=130)
    except Exception as exc:  # grafik isteğe bağlıdır
        print("Eğri grafiği yazılamadı:", exc)
    print(f"Yazıldı: {args.out} ve ..._egriler.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
