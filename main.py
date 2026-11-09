"""
main.py - DIEM VAO CHUONG TRINH
  python main.py                 chay that: webcam + Arduino
  python main.py --sim           webcam that, Arduino GIA LAP (chua co mach)
  python main.py --video f.mp4   phat lai video, Arduino gia lap
  python main.py --selftest      tu kiem tra toan bo chuoi bang canh gia lap (khong can phan cung)
Phim: b chup nen | r chon ROI | k hieu chuan mm/px (dat vat tron KNOWN_D_MM vao buong)
      1/2/3 nhap loai THAT cua qua vua phan loai | g/x ket qua co khi DUNG/SAI thung
      t cua so chinh thong so | w luu thong so | s luu anh ROI (cho color_survey) | q thoat va in bao cao
"""
import argparse, os, queue, sys, time
import cv2
import numpy as np
import config, vision
from evaluation import Recorder, metrics_text
from hardware import GateController, SerialLink, SimLink
from pipeline import System
from sources import CameraSource, SyntheticSource, VideoSource


def build(args, source=None, link=None, headless=False, run_dir=None):
    rec = Recorder(run_dir)
    import threading
    stop_ev = threading.Event()
    link = link or (SimLink() if (args.sim or args.video) else SerialLink())
    gates = GateController(link, rec, stop_ev)
    P = vision.Params()
    src = source or (VideoSource(args.video) if args.video else CameraSource())
    system = System(src, gates, rec, P, headless=headless)
    system.stop_ev = stop_ev                      # dung chung mot co dung cho moi luong
    link.start()
    gates.start()
    system.start()
    return system, rec, gates, link


def tuning_window(P):
    cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
    def tb(name, init, maxv, fn):
        cv2.createTrackbar(name, "Tuning", int(init), maxv, fn)
    tb("fg_thresh", P.fg_thresh, 80, lambda v: setattr(P, "fg_thresh", v))
    tb("canny_lo", P.canny_lo, 200, lambda v: setattr(P, "canny_lo", v))
    tb("canny_hi", P.canny_hi, 300, lambda v: setattr(P, "canny_hi", v))
    tb("defect_%x10", P.defect_ratio * 1000, 100, lambda v: setattr(P, "defect_ratio", v / 1000.0))
    tb("de_min", P.defect_de_min, 40, lambda v: setattr(P, "defect_de_min", float(max(v, 1))))
    tb("k_x10", P.defect_k * 10, 80, lambda v: setattr(P, "defect_k", max(v, 5) / 10.0))
    tb("blob_mm2", P.defect_blob_mm2, 100, lambda v: setattr(P, "defect_blob_mm2", float(max(v, 1))))
    tb("H_lo", P.green_lo[0], 179, lambda v: P.green_lo.__setitem__(0, v))
    tb("H_hi", P.green_hi[0], 179, lambda v: P.green_hi.__setitem__(0, v))
    tb("S_lo", P.green_lo[1], 255, lambda v: P.green_lo.__setitem__(1, v))
    tb("V_lo", P.green_lo[2], 255, lambda v: P.green_lo.__setitem__(2, v))


def run_interactive(args):
    run_dir = os.path.join("runs", (args.run_name or "run") + "_" + time.strftime("%Y%m%d_%H%M%S"))
    system, rec, gates, link = build(args, run_dir=run_dir)
    tuning = False
    print(__doc__)
    try:
        while True:
            try:
                vis, dbg = system.disp_q.get(timeout=0.05)
                cv2.imshow("Phan loai tao xanh", vis)
                if dbg.get("edges") is not None:
                    cv2.imshow("Canny", dbg["edges"])
                if dbg.get("fg") is not None:
                    cv2.imshow("Phan nguong (fg)", dbg["fg"])
                if dbg.get("heat") is not None:                      # ban do lech mau: cang do cang bat thuong
                    cv2.imshow("Heat khuyet tat", cv2.applyColorMap(dbg["heat"], cv2.COLORMAP_JET))
            except queue.Empty:
                pass
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("b"):
                print("[NEN] da chup nen" if system.capture_background() else "[NEN] chua co anh")
            elif k == ord("r") and system.latest_frame is not None:
                x, y, w, h = cv2.selectROI("Chon ROI (keo chuot, Enter xac nhan)", system.latest_frame, False)
                cv2.destroyWindow("Chon ROI (keo chuot, Enter xac nhan)")
                if w > 10 and h > 10:
                    system.set_roi((x, y, x + w, y + h))
                    print(f"[ROI] {system.roi}  mm/px={system.mmpp:.4f}  (nho chup lai nen 'b')")
            elif k == ord("k"):
                m = system.calibrate_scale()
                print(f"[HIEU CHUAN] {m:.4f} mm/px" if m else "[HIEU CHUAN] chua thay vat mau trong buong")
            elif k in (ord("1"), ord("2"), ord("3")):
                print("[NHAN] loai that cho qua #", rec.set_truth(int(chr(k))))
            elif k == ord("g"):
                print("[CO KHI] DUNG thung, qua #", rec.set_outcome(True))
            elif k == ord("x"):
                print("[CO KHI] SAI thung, qua #", rec.set_outcome(False))
            elif k == ord("t") and not tuning:
                tuning_window(system.P); tuning = True
            elif k == ord("w"):
                system.P.save(); print("[THONG SO] da luu params.json")
            elif k == ord("s") and system.latest_roi is not None:
                fn = f"snap_{int(time.time())}.png"
                cv2.imwrite(fn, system.latest_roi); print("[ANH] da luu", fn)
    except KeyboardInterrupt:
        pass
    finally:
        system.stop(); gates.safe(); time.sleep(0.1); link.stop(); cv2.destroyAllWindows()
        rec.save()
        print(rec.summary())
        print("Da luu ket qua vao", run_dir)


def selftest(duration=27.0):
    """Tu kiem tra: canh gia lap 7 qua tao (co bong do hinh cau; vet dap nhat / vua / bang keo den) + Arduino gia lap. Kiem tra phan loai va lenh can gat."""
    config.BELT_SPEED_MM_S, config.OPEN_LEAD_S = 80.0, 0.3
    W, H, MMPP = 366, 333, 0.3
    config.ROI_LEN_CM = W * MMPP / 10.0
    spec = [(55, False, 1), (38, False, 2), (55, "moderate", 3), (36, "subtle", 3), (52, False, 1), (42, False, 2), (48, "tape", 3)]
    apples = [dict(d_mm=d, defect=df, t0=1.6 + i * 2.5) for i, (d, df, _) in enumerate(spec)]
    expected = [c for _, _, c in spec]

    class A: sim, video, run_name = True, None, None
    link = SimLink()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for f in ("calib.json", "background.png"):
        if os.path.exists(f): os.remove(f)
    system, rec, gates, link = build(A, SyntheticSource(apples, MMPP, W, H, 80.0), link, headless=True)
    system.P = vision.Params("__none__.json")
    time.sleep(0.8)
    system.capture_background()
    time.sleep(duration)
    system.stop(); gates.safe(); time.sleep(0.1)
    for r, e in zip(rec.results, expected):
        r["truth"] = e                      # gan nhan that theo thu tu qua tao
    print(rec.summary())
    got = [r["pred"] for r in rec.results]
    cmds = "".join(c for _, c in link.log)
    ok_cls = got == expected
    ok_gate = ("A" in cmds and "B" in cmds and cmds.count("A") == cmds.count("a") and
               cmds.count("B") == cmds.count("b") and not any("XUNG DOT" in r["warn"] for r in rec.results))
    print("\nLoai du doan :", got, "\nLoai ky vong :", expected)
    print("Lenh gui Arduino:", cmds)
    print("KET QUA TU KIEM TRA:", "PASS" if (ok_cls and ok_gate) else "FAIL",
          f"(phan loai {'dung' if ok_cls else 'SAI'}, lich can gat {'dung' if ok_gate else 'SAI'})")
    return ok_cls and ok_gate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true"); ap.add_argument("--video")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run-name")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    run_interactive(a)
