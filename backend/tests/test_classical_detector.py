import cv2
import numpy as np

from app.services.image_processing import ImageProcessingService


def _scene(seed=0, shape=(240, 320)):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 6, shape).astype(np.float32)
    bg = 70 + cv2.GaussianBlur(noise, (0, 0), 3) * 3
    return np.clip(bg, 0, 255).astype(np.uint8)


def _add_person(img, x, y, w=22, h=70):
    """Sıcak, dikey, baş+gövde şekilli yapay leke (yalnızca birim testi içindir)."""
    cv2.ellipse(img, (x + w // 2, y + h // 2 + 8), (w // 2, h // 2 - 8), 0, 0, 360, 210, -1)
    cv2.circle(img, (x + w // 2, y + 8), 8, 225, -1)
    return (x, y, w, h)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float(aw * ah + bw * bh - inter)


def test_detects_upright_warm_blob():
    img = _scene()
    truth = _add_person(img, 150, 100)
    candidates, _ = ImageProcessingService().detect_people(img)
    assert any(_iou((c["x"], c["y"], c["w"], c["h"]), truth) > 0.4 for c in candidates)


def test_separates_two_people_side_by_side():
    img = _scene(1)
    a = _add_person(img, 110, 110)
    b = _add_person(img, 210, 105)
    candidates, _ = ImageProcessingService().detect_people(img)
    hits = [any(_iou((c["x"], c["y"], c["w"], c["h"]), t) > 0.4 for c in candidates) for t in (a, b)]
    assert all(hits)


def test_wide_horizontal_object_is_not_a_candidate():
    img = _scene(2)
    cv2.ellipse(img, (160, 150), (45, 16), 0, 0, 360, 215, -1)   # yatay, hayvan/araç benzeri
    candidates, _ = ImageProcessingService().detect_people(img)
    assert candidates == []


def test_colorbar_like_overlay_is_rejected():
    img = _scene(3)
    bar = np.linspace(255, 0, 90).astype(np.uint8).reshape(-1, 1).repeat(14, axis=1)
    img[15:105, 285:299] = bar
    img[15:50, 285:299] = 255
    img[75:105, 285:299] = 0
    candidates, _ = ImageProcessingService().detect_people(img)
    assert all(c["x"] + c["w"] < 280 or c["x"] > 300 for c in candidates)


def test_flat_image_has_no_candidates():
    candidates, _ = ImageProcessingService().detect_people(np.full((200, 300), 90, np.uint8))
    assert candidates == []


def test_accepts_16bit_and_color_input():
    base = _scene(4)
    truth = _add_person(base, 120, 90)
    svc = ImageProcessingService()
    for variant in ((base.astype(np.uint16) * 200 + 20000), cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)):
        candidates, _ = svc.detect_people(variant)
        assert any(_iou((c["x"], c["y"], c["w"], c["h"]), truth) > 0.4 for c in candidates)
