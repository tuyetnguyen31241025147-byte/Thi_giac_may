"""sources.py - NGUON ANH: webcam that, file video, va canh bang tai gia lap (de tu kiem tra)."""
import os, time, math, random
import cv2
import numpy as np
import config


class CameraSource:
    """Webcam USB (Sensor + Interface). Co the mo lai khi mat ket noi."""
    def __init__(self, index=config.CAM_INDEX, width=config.CAM_WIDTH, height=config.CAM_HEIGHT):
        self.index, self.width, self.height, self.cap = index, width, height, None

    def open(self):
        self.release()
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(self.index)
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if config.LOCK_CAMERA_AUTO:               # khong phai webcam nao cung ho tro; bo qua neu loi
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual (DirectShow)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAMERA] {w}x{h} @ {self.cap.get(cv2.CAP_PROP_FPS):.0f} fps (yeu cau {self.width}x{self.height})")
        return True

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoSource:
    """Phat lai file video theo toc do goc (de thu code khi chua co mo hinh)."""
    def __init__(self, path, loop=True):
        self.path, self.loop, self.cap, self._t = path, loop, None, 0.0

    def open(self):
        self.cap = cv2.VideoCapture(self.path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._t = time.perf_counter()
        return self.cap.isOpened()

    def read(self):
        wait = self._t + 1.0 / self.fps - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        self._t = time.perf_counter()
        ok, f = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return ok, f

    def release(self):
        if self.cap: self.cap.release()


class SyntheticSource:
    """
    Canh bang tai gia lap: nen xanh dam, tao xanh tuoi chay tu trai sang phai, co qua co vet nau
    (vet chi hien o mot so frame de mo phong tao xoay). Dung cho --selftest.
    apples: list dict(d_mm, defect, t0)  t0 = giay (tinh tu luc bat dau) tao bat dau vao khung.
    """
    def __init__(self, apples, mm_per_px=0.3, w=366, h=333, speed_mm_s=80.0, fps=30.0, seed=1):
        self.apples, self.mmpp, self.w, self.h = apples, mm_per_px, w, h
        self.speed, self.fps = speed_mm_s, fps
        self.rng = np.random.RandomState(seed)
        self.n = 0

    def open(self):
        self.t_start = time.perf_counter(); self.n = 0
        return True

    def read(self):
        wait = self.t_start + self.n / self.fps - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        t = self.n / self.fps
        img = np.full((self.h, self.w, 3), (38, 72, 34), np.uint8)
        img = np.clip(img.astype(np.int16) + self.rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        for i, a in enumerate(self.apples):
            r = a["d_mm"] / self.mmpp / 2.0
            cx = -r + (t - a["t0"]) * self.speed / self.mmpp
            if cx < -r or cx > self.w + r:
                continue
            self._draw_apple(img, cx, self.h // 2, r, a["defect"])
        self.n += 1
        return True, img

    DEFECT_COLORS = {True: (30, 65, 110), "strong": (30, 65, 110),      # vet nau dam
                     "tape": (20, 20, 20),                              # bang keo den
                     "moderate": (62, 178, 122),                        # vet dap vua (~12 L*)
                     "subtle": (72, 196, 128)}                          # vet dap nhat (~6 L*)

    def _draw_apple(self, img, cx, cy, r, defect):
        """Ve qua tao co bong do hinh cau (giua sang, mep toi), vet loa sang, va (neu co) vet loi."""
        x0, x1 = max(0, int(cx - r) - 1), min(self.w, int(cx + r) + 2)
        y0, y1 = max(0, int(cy - r) - 1), min(self.h, int(cy + r) + 2)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
        m = d2 <= 1.0
        shade = (1.0 - 0.32 * d2)[..., None]
        patch = img[y0:y1, x0:x1]
        patch[m] = np.clip(np.array((85, 215, 135), np.float32) * shade[m], 0, 255).astype(np.uint8)
        cv2.circle(img, (int(cx - 0.35 * r), int(cy - 0.35 * r)), max(2, int(0.1 * r)), (235, 250, 240), -1)
        if defect and ((self.n // 6) % 3 != 0):                          # vet hien o ~2/3 so frame (qua xoay)
            col = self.DEFECT_COLORS[defect]
            ex, ey = int(cx + 0.25 * r), int(cy + 0.1 * r)
            # vet "an" vao mau da (nhan voi shade) de tranh nhin thay mep tron
            ov = img.copy()
            cv2.ellipse(ov, (ex, ey), (int(0.24 * r), int(0.19 * r)), 20, 0, 360, col, -1)
            img[...] = np.where((ov != img).any(axis=2)[..., None], np.clip(ov * (0.75 + 0.25 * shade.mean()), 0, 255).astype(np.uint8), img)

    def release(self):
        pass
