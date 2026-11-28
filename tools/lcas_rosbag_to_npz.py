#!/usr/bin/env python3
"""LCAS Thermal Physiological Monitoring veri setindeki bir rosbag'i inceler ve dönüştürür.

ROS kurulumu GEREKMEZ; yalnızca `rosbags` ve `numpy` paketleri kullanılır:

    pip install rosbags numpy

İki kullanım:

  1) Yalnızca bilgi (hiçbir şey yazmaz): konuları, mesaj türlerini, gerçek kare hızını ve
     zaman damgası tutarlılığını raporlar.
         python tools/lcas_rosbag_to_npz.py KAYIT.bag --info

  2) Dönüştürme: termal kareleri ham değerleriyle ve zaman damgalarıyla .npz dosyasına yazar.
         python tools/lcas_rosbag_to_npz.py KAYIT.bag --out kayit.npz [--topic /konu] [--max-frames N]

Sıcaklık dönüşümü (veri seti sayfasındaki formül): derece_C = (ham_deger - 1000) / 10.
Betik ham değerleri değiştirmeden saklar; formül yalnızca üst veriye (meta) yazılır ve °C'ye çevirme
analiz aşamasında yapılır.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

IMAGE_TYPE = "sensor_msgs/msg/Image"
DECODABLE = {"mono16": np.dtype("<u2"), "16UC1": np.dtype("<u2"),
             "mono8": np.dtype("u1"), "8UC1": np.dtype("u1"), "32FC1": np.dtype("<f4")}


def rate_stats(times_s):
    """Zaman damgası dizisinden kare hızı istatistikleri (saniye cinsinden damgalar)."""
    t = np.asarray(times_s, dtype=np.float64)
    if t.size < 3:
        return {"frames": int(t.size)}
    dt = np.diff(t)
    med = float(np.median(dt))
    return {
        "frames": int(t.size),
        "duration_s": round(float(t[-1] - t[0]), 3),
        "median_dt_s": round(med, 5),
        "median_rate_hz": round(1.0 / med, 3) if med > 0 else None,
        "mean_rate_hz": round(float((t.size - 1) / (t[-1] - t[0])), 3) if t[-1] > t[0] else None,
        "min_dt_s": round(float(dt.min()), 5),
        "max_dt_s": round(float(dt.max()), 5),
        "gaps_over_3x_median": int(np.sum(dt > 3 * med)),
        "non_monotonic": int(np.sum(dt <= 0)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="Girdi .bag dosyası")
    ap.add_argument("--info", action="store_true", help="Yalnızca rapor yaz, dönüştürme yapma")
    ap.add_argument("--out", help="Çıktı .npz yolu")
    ap.add_argument("--topic", help="Görüntü konusu (verilmezse tek görüntü konusu otomatik seçilir)")
    ap.add_argument("--max-frames", type=int, default=0, help="En çok bu kadar kare yaz (0 = hepsi)")
    args = ap.parse_args()
    if not args.info and not args.out:
        ap.error("--info ya da --out verilmeli")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(args.bag) as reader:
        print(f"Dosya: {args.bag}")
        print(f"Toplam süre: {reader.duration / 1e9:.2f} s | toplam mesaj: {reader.message_count}")
        print("Konular:")
        for c in reader.connections:
            print(f"  {c.topic:40s} {c.msgtype:35s} {c.msgcount} mesaj")
        images = [c for c in reader.connections if c.msgtype == IMAGE_TYPE]
        if args.topic:
            images = [c for c in images if c.topic == args.topic]
        if not images:
            print("\nHATA: sensor_msgs/Image türünde konu bulunamadı. Yukarıdaki listeyi bana gönderin.")
            return 2
        if len(images) > 1:
            print("\nBirden çok görüntü konusu var; --topic ile birini seçin:", [c.topic for c in images])
            return 2
        conn = images[0]

        bag_times, header_times = [], []
        frames = None  # ilk mesajda, mesaj sayısına göre bir kez ayrılır (bellek dostu)
        written = 0
        meta = {}
        for _, ts, raw in reader.messages(connections=[conn]):
            msg = typestore.deserialize_ros1(raw, conn.msgtype)
            if not meta:
                meta = {"width": int(msg.width), "height": int(msg.height), "encoding": str(msg.encoding),
                        "step": int(msg.step), "is_bigendian": int(msg.is_bigendian)}
                print(f"\nGörüntü: {meta['width']}x{meta['height']}, kodlama={meta['encoding']}, "
                      f"bigendian={meta['is_bigendian']}")
            bag_times.append(ts / 1e9)
            header_times.append(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)
            if args.info:
                continue
            if meta["encoding"] not in DECODABLE:
                print(f"HATA: desteklenmeyen kodlama '{meta['encoding']}'. Bu çıktıyı bana gönderin.")
                return 2
            dtype = DECODABLE[meta["encoding"]].newbyteorder(">" if meta["is_bigendian"] else "<")
            frame = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(meta["height"], -1)[:, :meta["width"]]
            if frames is None:
                total = conn.msgcount if not args.max_frames else min(conn.msgcount, args.max_frames)
                frames = np.empty((total, meta["height"], meta["width"]), dtype=frame.dtype)
            frames[written] = frame
            written += 1
            if written >= len(frames):
                break

    print("\nKare hızı (rosbag kayıt zamanına göre):")
    print(json.dumps(rate_stats(bag_times), indent=2))
    print("Kare hızı (mesaj başlığındaki header.stamp'e göre):")
    print(json.dumps(rate_stats(header_times), indent=2))
    if args.info:
        if meta:
            print("\nBu rapor hiçbir dosya yazmadı. Çıktıyı bana yapıştırın.")
        return 0

    out = Path(args.out)
    frames = frames[:written]
    np.savez_compressed(
        out, frames=frames, bag_time_s=np.asarray(bag_times[:written]),
        header_time_s=np.asarray(header_times[:written]),
        meta=json.dumps({**meta, "topic": conn.topic, "celsius_formula": "(raw - 1000) / 10",
                         "source_bag": Path(args.bag).name}),
    )
    print(f"\nYazıldı: {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} kare)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
