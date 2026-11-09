"""
evaluation.py - GHI SO LIEU THUC NGHIEM VA TINH CHI SO DANH GIA
  - Moi qua tao: loai du doan, loai that (phim 1/2/3), ket qua co khi (phim g/x), d_mm, so frame, do tre...
  - Moi frame: thoi gian xu ly (ms); FPS; do tre lenh Arduino (ms)
Chay rieng:  python evaluation.py runs/<ten>/results.csv [--gt ground_truth.csv]
  ground_truth.csv: id,true_class,true_d_mm  (neu muon tinh sai so kich thuoc so voi thuoc kep)
"""
import csv, os, sys, threading, time
import numpy as np

FIELDS = ["id", "pred", "truth", "outcome", "d_mm", "area_mm2", "defect_max", "defect_blob_mm2", "n_frames", "n_def_frames",
          "track_s", "latency_ms", "warn"]


class Recorder:
    def __init__(self, run_dir=None):
        self.run_dir = run_dir
        self.lock = threading.Lock()
        self.results, self.proc_ms, self.frame_t, self.ack_ms, self.events = [], [], [], [], []
        self.n_captured = self.n_processed = 0
        self.t0 = time.perf_counter()

    def log_event(self, msg):
        self.events.append(f"{time.perf_counter() - self.t0:8.2f}s  {msg}")

    def add_result(self, **kw):
        with self.lock:
            kw["id"] = len(self.results) + 1
            kw.setdefault("truth", ""); kw.setdefault("outcome", ""); kw.setdefault("warn", "")
            self.results.append(kw)
            return kw["id"]

    def set_truth(self, cls):
        with self.lock:
            for r in reversed(self.results):
                if r["truth"] == "":
                    r["truth"] = cls
                    return r["id"]

    def set_outcome(self, ok):
        with self.lock:
            for r in reversed(self.results):
                if r["outcome"] == "":
                    r["outcome"] = "OK" if ok else "SAI"
                    return r["id"]

    def fps_now(self, n=30):
        t = self.frame_t[-n:]
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] else 0.0

    def save(self):
        if not self.run_dir:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader(); w.writerows(self.results)
        np.savetxt(os.path.join(self.run_dir, "proc_ms.csv"), np.array(self.proc_ms), fmt="%.3f", header="proc_ms")
        open(os.path.join(self.run_dir, "events.log"), "w", encoding="utf-8").write("\n".join(self.events))
        open(os.path.join(self.run_dir, "summary.txt"), "w", encoding="utf-8").write(self.summary())

    def summary(self):
        lines = ["=" * 56, "KET QUA DANH GIA", "=" * 56]
        if self.frame_t and len(self.frame_t) > 2:
            lines.append(f"FPS camera trung binh   : {(len(self.frame_t) - 1) / (self.frame_t[-1] - self.frame_t[0]):.1f}")
        if self.n_captured:
            lines.append(f"Frame xu ly / thu duoc  : {self.n_processed}/{self.n_captured} "
                         f"(bo {100 * (1 - self.n_processed / self.n_captured):.1f}% frame cu de giu real-time)")
        if self.proc_ms:
            p = np.array(self.proc_ms)
            lines.append(f"Thoi gian xu ly / frame : TB {p.mean():.1f} ms | trung vi {np.median(p):.1f} | "
                         f"P95 {np.percentile(p, 95):.1f} | max {p.max():.1f} ms")
        if self.ack_ms:
            a = np.array(self.ack_ms)
            lines.append(f"Do tre lenh -> Arduino ACK: TB {a.mean():.1f} ms | max {a.max():.1f} ms (n={len(a)})")
        lines += metrics_text(self.results)
        return "\n".join(lines)


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def metrics_text(rows, gt=None):
    """rows: danh sach dict ket qua (tu Recorder hoac doc CSV). gt: {id: (true_class, true_d_mm)}."""
    out = []
    if not rows:
        return ["Chua co qua tao nao duoc ghi nhan."]
    out.append(f"So qua tao da phan loai : {len(rows)}")
    lat = [x for x in (_f(r.get("latency_ms")) for r in rows) if x is not None]
    if lat:
        out.append(f"Do tre camera -> quyet dinh: TB {np.mean(lat):.1f} ms")
    nf = [x for x in (_f(r.get("n_frames")) for r in rows) if x is not None]
    if nf:
        out.append(f"So frame hop le / qua    : TB {np.mean(nf):.1f} (min {min(nf):.0f})")
    # --- do chinh xac phan loai ---
    labeled = []
    for r in rows:
        t = r.get("truth")
        if gt and int(r["id"]) in gt:
            t = gt[int(r["id"])][0]
        if t not in ("", None):
            labeled.append((int(float(t)), int(float(r["pred"]))))
    if labeled:
        cm = np.zeros((3, 3), int)
        for t, p in labeled:
            cm[t - 1][p - 1] += 1
        out.append(f"Do chinh xac phan loai  : {100 * np.trace(cm) / cm.sum():.1f}% ({np.trace(cm)}/{cm.sum()} qua)")
        out.append("Ma tran nham lan (hang = THAT, cot = DU DOAN)      Loai1 Loai2 Loai3")
        for i in range(3):
            out.append(f"   That Loai {i + 1}                                      {cm[i][0]:5d} {cm[i][1]:5d} {cm[i][2]:5d}")
        for i in range(3):
            tp = cm[i][i]
            pr = tp / cm[:, i].sum() if cm[:, i].sum() else float("nan")
            rc = tp / cm[i].sum() if cm[i].sum() else float("nan")
            out.append(f"   Loai {i + 1}: precision {pr * 100:5.1f}% | recall {rc * 100:5.1f}%")
    else:
        out.append("(chua co nhan loai THAT: bam phim 1/2/3 sau moi qua de tinh do chinh xac)")
    # --- ty le thanh cong co khi ---
    oc = [r.get("outcome") for r in rows if r.get("outcome") in ("OK", "SAI")]
    if oc:
        out.append(f"Ty le gat DUNG thung (co khi): {100 * oc.count('OK') / len(oc):.1f}% ({oc.count('OK')}/{len(oc)} qua)")
    # --- sai so kich thuoc ---
    if gt:
        err = [float(r["d_mm"]) - gt[int(r["id"])][1] for r in rows
               if int(r["id"]) in gt and gt[int(r["id"])][1] is not None and _f(r.get("d_mm"))]
        if err:
            e = np.array(err)
            out.append(f"Sai so duong kinh (do - thuoc kep): TB {e.mean():+.2f} mm | MAE {np.abs(e).mean():.2f} mm | "
                       f"RMSE {np.sqrt((e ** 2).mean()):.2f} mm (n={len(e)})")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
    gt = None
    if "--gt" in sys.argv:
        gt = {}
        for r in csv.DictReader(open(sys.argv[sys.argv.index("--gt") + 1], encoding="utf-8")):
            gt[int(r["id"])] = (int(r["true_class"]), _f(r.get("true_d_mm")))
    print("\n".join(metrics_text(rows, gt)))
