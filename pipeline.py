"""
pipeline.py - CHUOI THOI GIAN THUC:  Camera -> Xu ly anh -> Ra quyet dinh -> Dieu khien phan cung
DA LUONG (multithreading):
  [capture]  doc camera, chi giu frame MOI NHAT (bo frame cu de khong tre tich luy), tu mo lai khi mat camera
  [process]  loc/tach nen/Canny/contour -> theo doi qua -> gop nhieu frame -> quyet dinh -> len lich can gat
  [gates]    (hardware.GateController) hen gio va gui lenh mo/dong can qua Serial
  [serial]   (hardware.SerialLink) doc ACK tu Arduino, tu noi lai
  main       hien thi + phim bam (imshow bat buoc chay o luong chinh)
"""
import json, os, queue, threading, time, traceback
import cv2
import numpy as np
import config, vision, tracker
from hardware import plan_for_apple

CLASS_NAME = {1: "Loai 1 (to)", 2: "Loai 2 (nho)", 3: "Loai 3 (khuyet tat)"}
COLOR = {1: (0, 200, 0), 2: (255, 160, 0), 3: (0, 0, 255)}
CALIB_FILE, BG_FILE = "calib.json", "background.png"


class System:
    def __init__(self, source, gates, rec, P, headless=False):
        self.source, self.gates, self.rec, self.P, self.headless = source, gates, rec, P, headless
        self.stop_ev = threading.Event()
        self.lock = threading.Lock()
        self.frame_q, self.disp_q = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.tracker = tracker.Tracker()
        self.roi, self.mmpp, self.bg = None, 0.2, None
        self.latest_frame = self.latest_roi = None
        self.last_d_px = None
        self.cam_ok, self.last_ok_t = False, time.perf_counter()
        self.warn = {"light": False, "focus": False}
        self.recent = []                   # vai quyet dinh gan nhat de hien thi
        self.threads = []

    # ---------------- hieu chuan ----------------
    def set_roi(self, roi):
        x0, y0, x1, y1 = [int(v) for v in roi]
        with self.lock:
            self.roi = (x0, y0, x1, y1)
            self.mmpp = config.ROI_LEN_CM * 10.0 / max(1, x1 - x0)     # ROI dai dung ROI_LEN_CM
            self.bg = None
            self.tracker.reset()
        self.save_calib()

    def capture_background(self):
        if self.latest_roi is None:
            return False
        roi = self.latest_roi.copy()
        with self.lock:
            self.bg = vision.Background(roi)
            self.tracker.reset()
        cv2.imwrite(BG_FILE, roi)
        return True

    def calibrate_scale(self, known_d_mm=config.KNOWN_D_MM):
        if not self.last_d_px:
            return None
        self.mmpp = known_d_mm / self.last_d_px
        self.save_calib()
        return self.mmpp

    def save_calib(self):
        json.dump({"roi": self.roi, "mmpp": self.mmpp}, open(CALIB_FILE, "w"))

    def load_calib(self):
        try:
            c = json.load(open(CALIB_FILE))
            self.roi, self.mmpp = tuple(c["roi"]), float(c["mmpp"])
            if os.path.exists(BG_FILE):
                self.bg = vision.Background(cv2.imread(BG_FILE))
            print(f"[CALIB] ROI={self.roi} mm/px={self.mmpp:.4f} nen={'co' if self.bg else 'chua'}")
        except Exception:
            pass

    # ---------------- khoi dong / dung ----------------
    def start(self):
        self.load_calib()
        if not self.source.open():
            print("[CAMERA] khong mo duoc nguon anh (se thu lai)")
        for fn, name in ((self._capture_loop, "capture"), (self._process_loop, "process")):
            t = threading.Thread(target=fn, daemon=True, name=name)
            t.start()
            self.threads.append(t)

    def stop(self):
        self.stop_ev.set()
        time.sleep(0.2)
        self.source.release()

    # ---------------- luong 1: camera ----------------
    def _capture_loop(self):
        while not self.stop_ev.is_set():
            ok, frame = self.source.read()
            now = time.perf_counter()
            if ok and frame is not None:
                if not self.cam_ok:
                    self.rec.log_event("camera OK")
                self.cam_ok, self.last_ok_t = True, now
                self.rec.n_captured += 1
                self.rec.frame_t.append(now)
                if self.frame_q.full():
                    try: self.frame_q.get_nowait()          # bo frame cu: uu tien frame moi nhat
                    except queue.Empty: pass
                self.frame_q.put((frame, now))
            else:
                time.sleep(0.02)
                if now - self.last_ok_t > config.CAM_TIMEOUT_S:
                    if self.cam_ok:
                        self.cam_ok = False
                        self.rec.log_event("MAT CAMERA")
                        self.gates.safe()                  # su co -> dua can ve trang thai an toan
                    self.source.open()                      # thu mo lai
                    self.last_ok_t = now - config.CAM_TIMEOUT_S + 1.0   # thu lai moi ~1 s

    # ---------------- luong 2: xu ly + quyet dinh ----------------
    def _process_loop(self):
        nframe = 0
        while not self.stop_ev.is_set():
            try:
                frame, t_cap = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._process_one(frame, t_cap, nframe)
            except Exception:
                traceback.print_exc()
            nframe += 1

    def _process_one(self, frame, t_cap, nframe):
        t0 = time.perf_counter()
        H, W = frame.shape[:2]
        if self.roi is None:
            self.set_roi((0, 0, W, H))
        x0, y0, x1, y1 = self.roi
        roi = frame[y0:y1, x0:x1]
        self.latest_frame, self.latest_roi = frame, roi
        vis = frame.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 1)
        decision_x = int((x1 - x0) * config.DECISION_X_FRAC)
        cv2.line(vis, (x0 + decision_x, y0), (x0 + decision_x, y1), (0, 255, 255), 1)

        P, bg, mmpp = self.P, self.bg, self.mmpp
        blobs, dbg = vision.analyze(roi, bg, P, mmpp)
        if blobs:
            self.last_d_px = max(b["d_px"] for b in blobs)

        newly = self.tracker.update(blobs, decision_x, config.MATCH_MM / mmpp)
        for b in blobs:
            t = b["track"]
            cls_f = vision.classify_frame(b, P)
            sub = vis[y0:y1, x0:x1]
            cv2.drawContours(sub, [b["cnt"]], -1, COLOR[cls_f], 2)
            cv2.drawContours(sub, b["dcnts"], -1, (0, 0, 255), 2)
            n_def = sum(t.flags)
            cv2.putText(vis, f"#{t.id} n={len(t.ratios)} loi={n_def} d={b['d_mm']:.0f}mm",
                        (x0 + int(b["cx"]) - 60, y0 + int(b["cy"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[cls_f], 1)

        for t in newly:
            d = tracker.decide(t, P)
            x_cm = config.ROI_X0_CM + t.cx * mmpp / 10.0
            r_cm = d["d_mm"] / 20.0 if d["n"] else 2.5
            now = time.perf_counter()
            warns = plan_for_apple(self.gates, d["cls"], x_cm, r_cm, t_cap, now)
            if d["low_conf"]:
                warns.append(f"it frame hop le ({d['n']})")
            rid = self.rec.add_result(pred=d["cls"], d_mm=round(d["d_mm"], 1), area_mm2=round(d["area"]),
                                      defect_max=round(d["r_max"], 4), defect_blob_mm2=round(d["blob_max"], 1), n_frames=d["n"], n_def_frames=d["n_def"],
                                      track_s=round(now - t.t_first, 2), latency_ms=round((now - t_cap) * 1000, 1),
                                      warn=" | ".join(warns))
            msg = (f"[#{rid}] {CLASS_NAME[d['cls']]}  d={d['d_mm']:.1f}mm  A={d['area']:.0f}mm2  "
                   f"frame={d['n']}  frame_loi={d['n_def']}  loi_max={d['r_max'] * 100:.1f}%  vung_loi_max={d['blob_max']:.0f}mm2")
            print(msg + ("   !! " + "; ".join(warns) if warns else ""))
            self.recent = (self.recent + [(rid, d["cls"])])[-4:]

        if nframe % 10 == 0 and bg is not None and dbg.get("fg") is not None:      # giam sat chat luong anh
            focus, vmean = vision.quality(roi, dbg["fg"])
            self.warn["light"] = abs(vmean - bg.v_mean) / max(bg.v_mean, 1) > config.LIGHT_TOL
            self.warn["focus"] = focus < bg.focus * config.FOCUS_TOL

        ms = (time.perf_counter() - t0) * 1000
        self.rec.proc_ms.append(ms)
        self.rec.n_processed += 1
        self._overlay(vis, ms)
        if self.disp_q.full():
            try: self.disp_q.get_nowait()
            except queue.Empty: pass
        self.disp_q.put((vis, dbg))

    def _overlay(self, vis, ms):
        g = self.gates.status()
        lines = [f"Camera: {'OK' if self.cam_ok else 'MAT CAMERA!'} | {self.rec.fps_now():.0f} FPS | xu ly {ms:.1f} ms",
                 f"Arduino: {'OK' if g['link_ok'] else 'MAT KET NOI!'} ({g['name']}) | "
                 f"Can1 {'MO' if g['state'][1] else 'dong'} | Can2 {'MO' if g['state'][2] else 'dong'}"]
        if self.recent:
            lines.append("Gan nhat: " + "  ".join(f"#{i}->L{c}" for i, c in self.recent))
        if self.bg is None:
            lines.append("CHUA CO ANH NEN: de buong trong roi nhan 'b'")
        if self.warn["light"]:
            lines.append("CANH BAO: anh sang thay doi -> nhan 'b' chup lai nen")
        if self.warn["focus"]:
            lines.append("CANH BAO: anh mo (mat net / ban ong kinh)")
        for i, s in enumerate(lines):
            bad = "CANH BAO" in s or "!" in s or "CHUA" in s
            cv2.putText(vis, s, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if bad else (255, 255, 255), 2)
