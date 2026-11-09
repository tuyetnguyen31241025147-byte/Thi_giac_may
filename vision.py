"""
vision.py - XU LY ANH (chi dung thuat toan truyen thong, KHONG machine learning)

Quy trinh theo yeu cau thay:
  B0. Khao sat khong gian mau        -> tools: color_survey.py (chon Lab/HSV dua tren diem Fisher)
  B1. Phan nguong                    -> tru nen trong khong gian Lab (hoac xam) + threshold
  B2. Nhan dien bien (Canny)         -> tim canh nut/dap/vet sau ben trong qua
  B3. Contour (phat hien khuyet tat) -> gop vung loi (cat mau HSV + canh Canny) -> contour -> ty le loi
Ky thuat bo sung: Median filter, Gaussian blur, hinh thai hoc (Opening/Closing), bao loi (convex hull),
cat mau HSV (Color Slicing), mat na co (erode), toan tu Laplacian do do net anh.
PHAT HIEN KHUYET TAT = gop 3 tin hieu: (1) lech mau so voi "da chuan" cua CHINH qua tao do (Lab, thich nghi, da hieu
chinh bong do hinh cau), (2) vung khong thuoc dai xanh HSV, (3) net Canny. Tin hieu (1) la thu bat duoc vet dap nhat.
Moi kich thuoc khoi tinh bang mm roi doi sang pixel -> khong phu thuoc do phan giai.
"""
import json, math, os
import cv2
import numpy as np
import config


def mm2px(mm, mmpp, odd=False, minimum=1):
    v = max(minimum, int(round(mm / mmpp)))
    if odd and v % 2 == 0:
        v += 1
    return v


class Params:
    """Cac thong so co the chinh luc dang chay (cua so Tuning) va luu vao params.json."""
    DEFAULTS = dict(
        bg_method=config.BG_METHOD, fg_thresh=config.FG_THRESH, canny_lo=config.CANNY_LO, canny_hi=config.CANNY_HI,
        defect_ratio=config.DEFECT_RATIO, strong_ratio=config.STRONG_RATIO, erode_mm=config.ERODE_MM,
        defect_de_min=config.DEFECT_DE_MIN, defect_k=config.DEFECT_K, defect_blob_mm2=config.DEFECT_BLOB_MM2,
        strong_blob_mm2=config.STRONG_BLOB_MM2,
        morph_mm=config.MORPH_MM, min_blob_mm2=config.MIN_BLOB_MM2, min_defect_mm2=config.MIN_DEFECT_MM2,
        area_threshold_mm2=config.AREA_THRESHOLD_MM2, min_defect_frames=config.MIN_DEFECT_FRAMES, skin_deg=config.SKIN_POLY_DEG,
        green_lo=list(config.GREEN_LO), green_hi=list(config.GREEN_HI))

    def __init__(self, path="params.json"):
        self.path = path
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)
        if os.path.exists(path):
            try:
                for k, v in json.load(open(path)).items():
                    if k in self.DEFAULTS:
                        setattr(self, k, v)
            except Exception as e:
                print("[PARAMS] khong doc duoc", path, e)

    def save(self):
        json.dump({k: getattr(self, k) for k in self.DEFAULTS}, open(self.path, "w"), indent=1)


class Background:
    """Anh nen (buong trong). Luu cac dac trung da loc de tru nhanh."""
    def __init__(self, roi):
        self.raw = roi.copy()
        self._cache = {}
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.focus = float(cv2.Laplacian(g, cv2.CV_64F).var())             # do net (dao ham bac 2)
        self.v_mean = float(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2].mean())

    def features(self, k):
        if k not in self._cache:
            blur = cv2.medianBlur(self.raw, k)
            self._cache[k] = (cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32),
                              cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY))
        return self._cache[k]


def _poly_features(u, v, deg):
    cols = [np.ones_like(u)]
    for i in range(1, deg + 1):
        for j in range(i + 1):
            cols.append(u ** (i - j) * v ** j)
    return np.stack(cols, axis=-1)


def skin_residual(lab_c, inner_c, d_px, P):
    """
    Mo hinh "da chuan" cua chinh qua tao do: khop MAT DA THUC (bac P.skin_deg) cho tung kenh Lab ben trong mat na,
    co loai ngoai lai lap lai (nhung diem lech nhieu bi bo khoi viec khop de vet loi khong keo mo hinh theo no).
    Mat da thuc bac 3 mo ta duoc bong do hinh cau (giua sang, mep toi) -> khong can nguong mau co dinh.
    Do lech mau (dist) = khoang cach Lab giua anh that va mo hinh. Nguong = max(DE_MIN, trung vi + K x MAD).
    Tra ve (mat na loi 0/255, ban do dist, trung vi, MAD) o kich thuoc anh cat.
    """
    h, w = inner_c.shape
    s = min(1.0, 100.0 / max(d_px, 1.0))              # lam viec o do phan giai ~100 px/qua cho nhanh
    if s < 1.0:
        lab_s = cv2.resize(lab_c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m_s = cv2.resize(inner_c, (lab_s.shape[1], lab_s.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        lab_s, m_s = lab_c, inner_c
    hs, ws = m_s.shape
    m = m_s > 0
    ys, xs = np.nonzero(m)
    if xs.size < 60:
        return np.zeros((h, w), np.uint8), np.zeros((h, w), np.float32), 0.0, 1.0
    cx0, cy0, rs = xs.mean(), ys.mean(), max(8.0, d_px * s / 2.0)
    yy, xx = np.mgrid[0:hs, 0:ws]
    F = _poly_features((xx - cx0) / rs, (yy - cy0) / rs, int(P.skin_deg)).reshape(-1, 1 + sum(range(2, int(P.skin_deg) + 2)))
    Y = lab_s.reshape(-1, 3).astype(np.float64)
    idx = np.flatnonzero(m.ravel())
    wgt = np.ones(idx.size)
    thr, med, mad = P.defect_de_min, 0.0, 1.0
    dist = np.zeros(hs * ws, np.float32)
    for _ in range(4):
        coef, *_r = np.linalg.lstsq(F[idx] * wgt[:, None], Y[idx] * wgt[:, None], rcond=None)
        est = F @ coef
        dist = np.sqrt(((Y - est) ** 2).sum(axis=1)).astype(np.float32)
        vals = dist[idx]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-3
        thr = max(P.defect_de_min, med + P.defect_k * mad)
        wgt = (vals < thr).astype(np.float64)          # loai diem bat thuong khoi viec khop
        if wgt.sum() < 40:
            break
    dist = dist.reshape(hs, ws)
    mask = ((dist >= thr) & m).astype(np.uint8) * 255
    if s < 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        dist = cv2.resize(dist, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask, dist, med, mad


def analyze(roi, bg, P, mmpp):
    """
    roi: anh BGR vung quan sat; bg: Background; P: Params; mmpp: mm tren pixel.
    Tra ve (blobs, debug). Moi blob la dict: cnt, cx, cy, touches, area_mm2, d_mm, d_px, defect_ratio, dcnts
    """
    if bg is None or roi.shape != bg.raw.shape:
        return [], {}
    H, W = roi.shape[:2]

    # --- Tien xu ly: loc trung vi (khu nhieu hat, giu bien) ---
    k = min(9, mm2px(config.MEDIAN_MM, mmpp, odd=True, minimum=3))
    blur = cv2.medianBlur(roi, k)

    # --- B1: tru nen + phan nguong (Intensity-level slicing) ---
    blab, bgray = bg.features(k)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    if P.bg_method == "lab":              # khoang cach mau Lab: nhay ca voi vet nau sat mep qua
        diff = np.clip(np.sqrt(((lab - blab) ** 2).sum(axis=2)), 0, 255).astype(np.uint8)
    else:
        diff = cv2.absdiff(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), bgray)
    _, fg = cv2.threshold(diff, P.fg_thresh, 255, cv2.THRESH_BINARY)
    mk = mm2px(P.morph_mm, mmpp, odd=True, minimum=3)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kern)       # xoa dom nhieu
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kern)      # lap lo trong qua
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_blob_px = P.min_blob_mm2 / (mmpp * mmpp)
    cands = [c for c in cnts if cv2.contourArea(c) >= min_blob_px]
    debug = {"diff": diff, "fg": fg, "edges": None, "heat": np.zeros((H, W), np.uint8)}
    if not cands:
        return [], debug

    # --- B2: Canny (Gaussian blur -> gradient -> non-max suppression -> nguong kep) ---
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), P.canny_lo, P.canny_hi)
    debug["edges"] = edges

    # --- khong gian HSV: cat mau xanh va loai vet loa sang ---
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array(P.green_lo, np.uint8), np.array(P.green_hi, np.uint8))
    highlight = cv2.inRange(hsv, np.array(config.HIGHLIGHT_LO, np.uint8), np.array(config.HIGHLIGHT_HI, np.uint8))
    hl_big = cv2.dilate(highlight, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    not_green_not_hl = cv2.bitwise_and(cv2.bitwise_not(green), cv2.bitwise_not(highlight))

    er = max(1, mm2px(P.erode_mm, mmpp))
    ekern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1, 2 * er + 1))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    min_def_px = max(5.0, P.min_defect_mm2 / (mmpp * mmpp))

    blobs = []
    for c in cands:
        hull = cv2.convexHull(c)           # tao loi: lap phan khuyet o mep do vet toi giong mau bang
        a_px = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(hull)
        touches = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        M = cv2.moments(hull)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

        mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        inner = cv2.erode(mask, ekern)     # bo vien qua (bong toi, anh xanh phan chieu tu bang)

        d_px = math.sqrt(4 * a_px / math.pi)
        sl = (slice(y, y + h), slice(x, x + w))           # chi xu ly khung bao quanh qua
        inner_c = inner[sl]

        # B3: vung loi = (1) lech mau so voi da chuan cua qua  OR (2) khong xanh  OR (3) canh Canny -> contour
        res_mask, dist, med, mad = skin_residual(lab[sl], inner_c, d_px, P)
        res_bad = cv2.bitwise_and(cv2.bitwise_and(res_mask, inner_c), cv2.bitwise_not(hl_big[sl]))
        color_bad = cv2.bitwise_and(inner_c, not_green_not_hl[sl])
        edge_in = cv2.bitwise_and(cv2.bitwise_and(edges[sl], inner_c), cv2.bitwise_not(hl_big[sl]))
        edge_in = cv2.morphologyEx(cv2.dilate(edge_in, k3), cv2.MORPH_CLOSE, k7)   # noi net dut
        dmask = cv2.bitwise_or(cv2.bitwise_or(res_bad, color_bad), edge_in)
        dmask = cv2.morphologyEx(cv2.morphologyEx(dmask, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k3)
        dc, _ = cv2.findContours(dmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x, y))
        dc = [d for d in dc if cv2.contourArea(d) >= min_def_px]
        areas = [cv2.contourArea(d) for d in dc]
        ratio = sum(areas) / max(cv2.countNonZero(inner), 1)
        blob_mm2 = max(areas) * mmpp * mmpp if areas else 0.0
        defect = ratio > P.defect_ratio or blob_mm2 >= P.defect_blob_mm2
        heat = np.clip(dist * 6.0, 0, 255).astype(np.uint8)
        debug["heat"][sl] = np.maximum(debug["heat"][sl], cv2.bitwise_and(heat, inner_c))

        blobs.append(dict(cnt=hull, cx=cx, cy=cy, touches=touches, area_mm2=a_px * mmpp * mmpp,
                          d_mm=d_px * mmpp, d_px=d_px, defect_ratio=ratio, defect_blob_mm2=blob_mm2,
                          defect=defect, dcnts=dc))
    return blobs, debug


def classify_frame(b, P):
    """Ket qua cua RIENG mot frame (chi de hien thi; quyet dinh that su gop nhieu frame trong tracker.py)."""
    if b["defect"]:
        return 3
    return 1 if b["area_mm2"] >= P.area_threshold_mm2 else 2


def quality(roi, fg):
    """Giam sat chat luong anh: do net (phuong sai Laplacian) va do sang vung nen (kenh V)."""
    focus = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2]
    m = fg == 0
    return focus, float(v[m].mean()) if m.any() else float(v.mean())
