"""tracker.py - THEO DOI TUNG QUA VA GOP KET QUA NHIEU FRAME (qua tao lan/xoay -> thay nhieu mat)."""
import math, time
import numpy as np
import config


class Track:
    _next = 1

    def __init__(self, frame_no):
        self.id = Track._next
        Track._next += 1
        self.cx = self.cy = 0.0
        self.last = frame_no
        self.t_first = time.perf_counter()
        self.areas, self.dmm, self.ratios, self.flags, self.blobs = [], [], [], [], []
        self.decided = False


class Tracker:
    def __init__(self):
        self.tracks, self.n = [], 0

    def reset(self):
        self.tracks, self.n = [], 0

    def update(self, blobs, decision_x_px, match_px):
        """Ghep blob vao qua dang theo doi (gan nhat). Tra ve danh sach qua VUA vuot vach quyet dinh."""
        self.n += 1
        taken, decided = set(), []
        for b in blobs:
            best, bd = None, match_px
            for t in self.tracks:
                if t.id in taken:
                    continue
                d = math.hypot(b["cx"] - t.cx, b["cy"] - t.cy)
                if d < bd:
                    best, bd = t, d
            if best is None:
                best = Track(self.n)
                self.tracks.append(best)
            taken.add(best.id)
            best.cx, best.cy, best.last = b["cx"], b["cy"], self.n
            if not b["touches"]:                  # chi lay so do khi qua nam tron trong khung
                best.areas.append(b["area_mm2"])
                best.dmm.append(b["d_mm"])
                best.ratios.append(b["defect_ratio"])
                best.flags.append(b["defect"])               # frame nay co bao loi khong
                best.blobs.append(b["defect_blob_mm2"])
            b["track"] = best
            if not best.decided and best.cx >= decision_x_px:
                best.decided = True
                decided.append(best)
        self.tracks = [t for t in self.tracks if self.n - t.last <= config.LOST_FRAMES]
        return decided


def decide(t, P):
    """
    Gop ket qua nhieu frame cua mot qua:
      - Khuyet tat: >= min_defect_frames frame bao loi, hoac 1 frame loi rat nang -> Loai 3
      - Kich thuoc: TRUNG VI dien tich qua cac frame -> Loai 1 / Loai 2
    """
    n = len(t.ratios)
    if n == 0:     # khong co frame hop le (qua to hon khung / chi thay mot phan) -> khong gat, can kiem tra tay
        return dict(cls=3, n=0, n_def=0, r_max=0.0, blob_max=0.0, d_mm=0.0, area=0.0, low_conf=True)
    n_def = sum(t.flags)
    r_max, blob_max = max(t.ratios), max(t.blobs)
    area, d_mm = float(np.median(t.areas)), float(np.median(t.dmm))
    if n_def >= min(P.min_defect_frames, n) or r_max > P.strong_ratio or blob_max >= P.strong_blob_mm2:
        cls = 3
    else:
        cls = 1 if area >= P.area_threshold_mm2 else 2
    return dict(cls=cls, n=n, n_def=n_def, r_max=r_max, blob_max=blob_max, d_mm=d_mm, area=area, low_conf=n < config.MIN_FRAMES)
