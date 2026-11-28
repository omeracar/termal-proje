#!/usr/bin/env python3
"""LCAS veri setindeki tüm kişiler için dönüştür → doğrula → olay tetikli test; özet tablo üretir.

Her kişi için: rosbag (.bag) bulunur, .npz'ye dönüştürülür, solunum doğrulaması (otomatik ROI) ve
olay tetikli tanı testi çalıştırılır. Sonuçlar OUT/<kisi>/ altına, özet OUT/lcas_summary.json'a yazılır.
Çok yer kaplayan .npz dosyaları işlem bitince silinir (--keep-npz ile saklanabilir).

  python scripts/lcas_batch.py --bags ~/Downloads/LCAS_Physiological_Monitoring_Dataset_rosbagfiles \
      --truth ~/Downloads/Ground_Truth --out ../data/validation/lcas
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-(.+)_respiration\.csv$")


def find_bag(bag_dir: Path, stamp: str, name: str):
    candidates = [p for p in bag_dir.rglob("*.bag") if p.name.startswith(stamp) and name.lower() in p.name.lower()]
    thermal = [p for p in candidates if "thermal" in p.name.lower()]
    return (thermal or candidates or [None])[0]


def run(cmd):
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        print("   HATA:", (result.stderr or result.stdout).strip().splitlines()[-1:] or result.returncode)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=4)
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--keep-npz", action="store_true")
    ap.add_argument("--only", nargs="*", help="Yalnızca bu kişi adları (örn. Claudio Manuel)")
    args = ap.parse_args()

    bag_dir, truth_dir, out_dir = Path(args.bags).expanduser(), Path(args.truth).expanduser(), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    summary = []
    for csv_path in sorted(truth_dir.glob("*_respiration.csv")):
        m = PATTERN.match(csv_path.name)
        if not m:
            continue
        stamp, name = m.groups()
        if args.only and name.lower() not in [o.lower() for o in args.only]:
            continue
        print(f"\n=== {name} ({stamp}) ===")
        bag = find_bag(bag_dir, stamp, name)
        if bag is None:
            print("   rosbag bulunamadı; atlandı")
            summary.append({"subject": name, "status": "rosbag_yok"})
            continue
        work = out_dir / name
        work.mkdir(exist_ok=True)
        npz = work / "kayit.npz"
        entry = {"subject": name, "bag": bag.name, "status": "ok"}
        if not run([py, ROOT / "tools" / "lcas_rosbag_to_npz.py", bag, "--out", npz]):
            entry["status"] = "donusturme_hatasi"
            summary.append(entry)
            continue
        val_json, trig_json = work / "validation.json", work / "triggered.json"
        run([py, ROOT / "backend" / "scripts" / "validate_breathing.py", "--npz", npz, "--reference", csv_path,
             "--roi", "auto", "--roi-preview", work / "roi.png", "--out", val_json, "--plot", work / "series.png"])
        run([py, ROOT / "backend" / "scripts" / "breathing_triggered.py", "--npz", npz, "--reference", csv_path,
             "--tile", args.tile, "--shuffles", args.shuffles, "--out", work / "triggered.png", "--json", trig_json])
        run([py, ROOT / "backend" / "scripts" / "breathing_compare.py", "--npz", npz, "--reference", csv_path,
             "--roi", "auto", "--out-dir", work / "compare"])
        cmp_json = work / "compare" / "compare.json"
        if cmp_json.exists():
            entry["compare"] = json.loads(cmp_json.read_text(encoding="utf-8"))
        if val_json.exists():
            v = json.loads(val_json.read_text(encoding="utf-8"))
            entry.update({"fps": v["sampling_rate_hz"], "roi": v["roi"], "validation": v["summary"]})
        if trig_json.exists():
            entry["triggered"] = json.loads(trig_json.read_text(encoding="utf-8"))
        if not args.keep_npz and npz.exists():
            npz.unlink()
        summary.append(entry)

    (out_dir / "lcas_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n\nÖZET")
    print(f"{'kişi':10s} {'fps':>6s} {'alg.oranı':>10s} {'ort.hata/dk':>12s} {'±3 içinde':>10s} {'tetikli p':>10s}")
    for e in summary:
        if e.get("status") != "ok" or "validation" not in e:
            print(f"{e['subject']:10s}  {e.get('status')}")
            continue
        v, t = e["validation"], e.get("triggered", {})
        fmt = lambda x: "—" if x is None else str(x)
        print(f"{e['subject']:10s} {e['fps']:6.1f} {fmt(v['detection_rate']):>10s} {fmt(v['mean_abs_error_per_min']):>12s} "
              f"{fmt(v['within_tolerance_fraction']):>10s} {fmt(t.get('p_fwer')):>10s}")
    print("\nYAPILANDIRMA KARŞILAŞTIRMASI (algılama oranı / ±3 içinde)")
    for e in summary:
        c = e.get("compare")
        if not c:
            continue
        print(f"{e['subject']}: sıçrama sayıları {c['steps']}, arka plan {c['background_steps']}, tekrar oranı {c['duplicate_fraction']}")
        for k, r in c["results"].items():
            print(f"   {k:32s} {r['detection_rate']} / {r['within_tolerance_fraction']}")
    print(f"\nAyrıntı: {out_dir / 'lcas_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
