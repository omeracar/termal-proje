"""
Dedektör başarım ölçümü: Precision, Recall, F1 ve görüntü başına işlem süresi.

Kullanım (backend klasöründen):
    python scripts/evaluate_detector.py --images ../data/samples --truth ../data/samples/ground_truth.json

ground_truth.json biçimi (kutular [x, y, genişlik, yükseklik], piksel cinsinden):
    {
      "1.jpg": [[120, 60, 25, 52], [165, 66, 22, 50]],
      "4.jpeg": []
    }
"_" ile başlayan anahtarlar yok sayılır. Etiketler gerçek görüntüler üzerinde elle çizilmelidir
(tools/label_boxes.html aracı, LabelImg/CVAT veya FLIR ADAS COCO etiketleri); bu betik etiket üretmez.
--vis KLASÖR verilirse her görüntü için yeşil=doğru, kırmızı=yanlış, sarı=kaçırılan kutulu resim yazılır.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.image_processing import ImageProcessingService  # noqa: E402


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float(aw * ah + bw * bh - inter)


def match_details(pred, truth, thr):
    """Skora göre azalan sırayla açgözlü, bire bir eşleştirme.
    (tp_pred_indeksleri, eşleşmeyen_gerçek_indeksleri) döner."""
    used = set()
    tp_idx = []
    order = sorted(range(len(pred)), key=lambda i: pred[i]["score"], reverse=True)
    for pi in order:
        p = pred[pi]
        box = (p["x"], p["y"], p["w"], p["h"])
        best, best_i = 0.0, -1
        for i, t in enumerate(truth):
            if i in used:
                continue
            v = iou(box, tuple(t))
            if v > best:
                best, best_i = v, i
        if best >= thr:
            used.add(best_i)
            tp_idx.append(pi)
    fn_idx = [i for i in range(len(truth)) if i not in used]
    return tp_idx, fn_idx


def match(pred, truth, thr):
    """(TP, FP, FN) döner."""
    tp_idx, fn_idx = match_details(pred, truth, thr)
    return len(tp_idx), len(pred) - len(tp_idx), len(fn_idx)


def draw_vis(image, pred, truth, thr):
    """TP yeşil, FP kırmızı, kaçırılan gerçek (FN) sarı kutu çizer."""
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image[:, :, :3].copy()
    if vis.dtype != "uint8":
        vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    tp_idx, fn_idx = match_details(pred, truth, thr)
    tp_set = set(tp_idx)
    for i, p in enumerate(pred):
        color = (0, 200, 0) if i in tp_set else (0, 0, 255)
        cv2.rectangle(vis, (p["x"], p["y"]), (p["x"] + p["w"], p["y"] + p["h"]), color, 2)
    for i in fn_idx:
        x, y, w, h = truth[i]
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 215, 255), 2)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="Görüntü klasörü")
    ap.add_argument("--truth", required=True, help="Etiket JSON dosyası")
    ap.add_argument("--iou", type=float, default=0.5, help="Eşleşme için en düşük IoU (varsayılan 0.5)")
    ap.add_argument("--out", help="Sonuçları JSON olarak kaydet")
    ap.add_argument("--vis", help="Bu klasöre görüntü başına kutulu sonuç resmi yaz (yeşil=doğru, kırmızı=yanlış, sarı=kaçırılan)")
    args = ap.parse_args()

    truth = {k: v for k, v in json.load(open(args.truth, encoding="utf-8")).items() if not k.startswith("_")}
    svc = ImageProcessingService()
    if args.vis:
        Path(args.vis).mkdir(parents=True, exist_ok=True)

    tot_tp = tot_fp = tot_fn = 0
    times, rows = [], []
    print(f"{'görüntü':20s}{'gerçek':>8s}{'bulunan':>9s}{'TP':>5s}{'FP':>5s}{'FN':>5s}{'süre(s)':>9s}")
    for name, boxes in sorted(truth.items()):
        path = Path(args.images) / name
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"{name:20s}  okunamadı, atlandı")
            continue
        t0 = time.perf_counter()
        pred, _ = svc.detect_people(image)
        dt = time.perf_counter() - t0
        tp, fp, fn = match(pred, boxes, args.iou)
        tot_tp, tot_fp, tot_fn = tot_tp + tp, tot_fp + fp, tot_fn + fn
        times.append(dt)
        if args.vis:
            cv2.imwrite(str(Path(args.vis) / f"{Path(name).stem}_sonuc.png"),
                        draw_vis(image, pred, boxes, args.iou))
        rows.append({"image": name, "truth": len(boxes), "predicted": len(pred),
                     "tp": tp, "fp": fp, "fn": fn, "seconds": round(dt, 3)})
        print(f"{name:20s}{len(boxes):8d}{len(pred):9d}{tp:5d}{fp:5d}{fn:5d}{dt:9.2f}")

    if not rows:
        sys.exit("Hiç görüntü değerlendirilemedi (yollar ve etiket dosyasını kontrol edin).")
    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    summary = {"images": len(rows), "iou_threshold": args.iou, "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
               "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
               "mean_seconds_per_image": round(sum(times) / len(times), 3)}
    print("\nTOPLAM:", json.dumps(summary, ensure_ascii=False))
    if args.out:
        json.dump({"summary": summary, "per_image": rows}, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
