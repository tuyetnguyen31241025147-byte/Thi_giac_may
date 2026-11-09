"""
hardware.py - GIAO TIEP ARDUINO VA DIEU KHIEN 2 CAN GAT

Co che can gat (thiet ke cua nhom): MAC DINH ca 2 can nam XEO chan ngang bang tai (DONG).
  Loai 1 : khong lam gi  -> tao truot vao can 1 -> truot theo can vao thung 1
  Loai 2 : MO can 1 (nam doc thanh bang) cho tao di qua, can 2 van DONG -> tao vao thung 2
  Loai 3 : MO ca 2 can -> tao di thang xuong thung 3 o cuoi bang
Giao thuc Serial 115200 (1 ky tu / lenh):
  PC -> Arduino : 'A' mo can 1 | 'a' dong can 1 | 'B' mo can 2 | 'b' dong can 2 | 'S' an toan (dong ca 2) | 'P' ping
  Arduino -> PC : "READY" khi khoi dong | "ACK:<lenh>" khi nhan lenh | "PONG" | "TIMEOUT:<can>" tu dong dong can
"""
import threading, time
import config
try:
    import serial
except ImportError:
    serial = None

CMD = {(1, True): "A", (1, False): "a", (2, True): "B", (2, False): "b"}


class BaseLink:
    def __init__(self):
        self.on_line, self.ok, self.name = None, False, "?"


class SerialLink(BaseLink):
    """Cong COM that toi Arduino. Tu dong noi lai khi rut cap."""
    def __init__(self, port=config.SERIAL_PORT, baud=config.SERIAL_BAUD):
        super().__init__()
        self.port, self.baud, self.name = port, baud, f"{port}@{baud}"
        self._ser, self._stop = None, threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def _open(self):
        if serial is None:
            print("[ARDUINO] chua cai pyserial (pip install pyserial)")
            return False
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.05, write_timeout=0.2)
            time.sleep(2.0)                    # mo cong lam Arduino reset, cho no khoi dong
            self.ok = True
            self._ser.write(b"P")
            return True
        except Exception as e:
            self.ok = False
            print(f"[ARDUINO] khong mo duoc {self.port}: {e}")
            return False

    def start(self):
        self._open()
        self._th.start()

    def write(self, ch):
        if not self.ok:
            return False
        try:
            self._ser.write(ch.encode())
            return True
        except Exception:
            self.ok = False
            return False

    def _loop(self):
        while not self._stop.is_set():
            if not self.ok:
                time.sleep(config.SERIAL_RECONNECT_S)
                self._open()
                continue
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
                if line and self.on_line:
                    self.on_line(line)
            except Exception:
                self.ok = False

    def stop(self):
        self._stop.set()
        try:
            if self._ser: self._ser.close()
        except Exception:
            pass


class SimLink(BaseLink):
    """Arduino gia lap (chay khi chua co phan cung): ghi nhat ky lenh va tra ACK sau ~8 ms."""
    def __init__(self):
        super().__init__()
        self.name, self.ok, self.log = "SIM", True, []

    def start(self):
        threading.Timer(0.05, lambda: self.on_line and self.on_line("READY")).start()

    def write(self, ch):
        self.log.append((time.perf_counter(), ch))
        threading.Timer(0.008, lambda: self.on_line and self.on_line(f"ACK:{ch}")).start()
        return True

    def stop(self):
        pass


class GateController(threading.Thread):
    """
    Luong DIEU KHIEN: giu danh sach cua so mo can (t_mo, t_dong) cho tung can; moi 3 ms so sanh
    trang thai mong muon voi trang thai da gui, chi gui lenh khi doi trang thai.
    """
    def __init__(self, link, rec, stop_ev):
        super().__init__(daemon=True, name="gates")
        self.link, self.rec, self.stop_ev = link, rec, stop_ev
        self.lock = threading.Lock()
        self.windows = {1: [], 2: []}
        self.state = {1: False, 2: False}          # trang thai MO da gui
        self.sent_t, self.ready, self._retry = {}, False, 0.0
        link.on_line = self._on_line

    # ---- len lich ----
    def add_window(self, g, t_open, t_close):
        with self.lock:
            w = self.windows[g]
            for i, (o, c) in enumerate(w):
                if t_open <= c + config.MERGE_GAP_S and t_close >= o - config.MERGE_GAP_S:
                    w[i] = (min(o, t_open), max(c, t_close))
                    return
            w.append((t_open, t_close))

    def open_during(self, g, t0, t1):
        with self.lock:
            return any(o <= t1 and c >= t0 for o, c in self.windows[g])

    # ---- vong lap chinh ----
    def run(self):
        while not self.stop_ev.is_set():
            now = time.perf_counter()
            if now >= self._retry:
                for g in (1, 2):
                    with self.lock:
                        self.windows[g] = [(o, c) for o, c in self.windows[g] if c > now - 2.0]
                        want = any(o <= now <= c for o, c in self.windows[g])
                    if want != self.state[g]:
                        self._send(g, want, now)
            time.sleep(0.003)

    def _send(self, g, open_, now):
        ch = CMD[(g, open_)]
        if self.link.write(ch):
            self.state[g] = open_
            self.sent_t[ch] = now
            self.rec.log_event(f"gui '{ch}' ({'MO' if open_ else 'DONG'} can {g})")
        else:
            self._retry = now + 0.2             # mat ket noi: thu lai sau 0,2 s

    def safe(self):
        """Tinh huong su co (mat camera...): huy lich, dong ca 2 can (trang thai mac dinh)."""
        with self.lock:
            self.windows = {1: [], 2: []}
        self.link.write("S")
        self.state = {1: False, 2: False}
        self.rec.log_event("AN TOAN: huy lich, dong ca 2 can")

    def _on_line(self, line):
        if line.startswith("ACK:") and len(line) >= 5:
            ch = line[4]
            if ch in self.sent_t:
                self.rec.ack_ms.append((time.perf_counter() - self.sent_t[ch]) * 1000)
        elif line == "READY":
            self.ready = True
            self.state = {1: False, 2: False}   # Arduino vua khoi dong: hai can dang DONG
            self.rec.log_event("Arduino READY")
        elif line == "PONG":
            self.ready = True
        elif line.startswith("TIMEOUT:"):
            try: self.state[int(line[8:])] = False
            except ValueError: pass
            self.rec.log_event("Arduino " + line)

    def status(self):
        return dict(state=dict(self.state), ready=self.ready, link_ok=self.link.ok, name=self.link.name)


def plan_for_apple(ctrl, cls, x_cm, r_cm, t_ref, now):
    """
    Tinh lich mo/dong can cho mot qua tao.
    x_cm : vi tri tam qua tren bang tai (cm) vao thoi diem t_ref (thoi diem chup frame quyet dinh)
    r_cm : ban kinh qua (cm).  Tra ve danh sach canh bao (rong neu on).
    """
    v = config.BELT_SPEED_MM_S / 10.0                       # cm/s
    warns = []
    t_at = lambda x, g: t_ref + (x - x_cm) / v + config.GATE_OFFSET_S[g]
    for g in (1, 2):
        P = config.GATE_PIVOT_CM[g]
        x_in = P - r_cm                                      # mep truoc cua qua cham dau can
        x_out = min(P + config.GATE_LEN_CM + r_cm + config.CLOSE_MARGIN_CM, config.BELT_LEN_CM + 1.0)
        need_open = (cls in (2, 3)) if g == 1 else (cls == 3)
        if need_open:
            t_o, t_c = t_at(x_in, g) - config.OPEN_LEAD_S, t_at(x_out, g)
            if t_o < now - 0.05:
                warns.append(f"can {g}: lenh mo tre {now - t_o:.2f}s (giam toc do bang hoac tang khoang cach)")
            ctrl.add_window(g, max(t_o, now), t_c)
        else:                                                # can phai DONG khi qua di toi
            if ctrl.open_during(g, t_at(x_in, g), t_at(P + config.GATE_LEN_CM / 2, g)):
                warns.append(f"can {g}: XUNG DOT lich (can dang mo cho qua truoc) - tao qua gan nhau")
    return warns
