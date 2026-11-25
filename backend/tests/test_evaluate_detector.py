import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_detector import draw_vis, iou, match  # noqa: E402


def P(x, y, w, h, score=1.0):
    return {"x": x, "y": y, "w": w, "h": h, "score": score}


def test_iou_identical_disjoint_and_half():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 10, 10)) - 1 / 3) < 1e-9


def test_match_perfect_and_empty_cases():
    assert match([P(0, 0, 10, 10)], [[0, 0, 10, 10]], 0.5) == (1, 0, 0)
    assert match([], [[0, 0, 10, 10]], 0.5) == (0, 0, 1)
    assert match([P(0, 0, 10, 10)], [], 0.5) == (0, 1, 0)
    assert match([], [], 0.5) == (0, 0, 0)


def test_match_is_one_to_one():
    # Aynı gerçek kutuya iki tahmin: yalnız biri TP, diğeri FP olmalı
    pred = [P(0, 0, 10, 10, 0.9), P(1, 0, 10, 10, 0.8)]
    assert match(pred, [[0, 0, 10, 10]], 0.5) == (1, 1, 0)


def test_match_threshold():
    pred = [P(0, 0, 10, 10)]
    truth = [[5, 0, 10, 10]]  # IoU = 1/3
    assert match(pred, truth, 0.5) == (0, 1, 1)
    assert match(pred, truth, 0.3) == (1, 0, 0)


def test_draw_vis_shape_and_colors():
    img = np.zeros((40, 40), dtype=np.uint8)
    vis = draw_vis(img, [P(2, 2, 10, 10), P(25, 25, 10, 10)], [[2, 2, 10, 10], [0, 20, 8, 8]], 0.5)
    assert vis.shape == (40, 40, 3)
    assert tuple(vis[2, 5]) == (0, 200, 0)      # TP yeşil
    assert tuple(vis[25, 28]) == (0, 0, 255)    # FP kırmızı
    assert tuple(vis[20, 3]) == (0, 215, 255)   # FN sarı
