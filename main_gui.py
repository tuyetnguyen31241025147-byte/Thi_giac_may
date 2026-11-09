import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
import os, queue
import argparse
import config

# Import nguyên vẹn hệ thống cũ của bạn
from main import build

# Tông màu
COLOR_NAVY = "#1A2B4C"
COLOR_WHITE = "#FFFFFF"
COLOR_GRAY = "#F0F2F5"
COLOR_TEXT = "#333333"

class AppleSorterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG PHÂN LOẠI TÁO TỰ ĐỘNG - VERSION GUI")
        self.root.geometry("1280x720")
        self.root.configure(bg=COLOR_GRAY)
        
        # Biến đếm sản lượng thực tế
        self.count_1, self.count_2, self.count_3 = 0, 0, 0
        self.last_result_count = 0
        self.conveyor_running = False

        self.setup_ui()

        # Gọi cái Lõi (hệ thống cũ) khởi động ngầm
        parser = argparse.ArgumentParser()
        parser.add_argument("--sim", action="store_true")
        parser.add_argument("--video")
        args = parser.parse_args()
        
        # Build hệ thống cũ với cờ headless = True (ẩn cửa sổ OpenCV cũ)
        self.system, self.rec, self.gates, self.link = build(args, headless=True)

        # Bắt đầu vòng lặp lấy hình ảnh
        self.update_frame()

    def setup_ui(self):
        # 1. KHU VỰC CAMERA (70%)
        self.left_frame = tk.Frame(self.root, bg=COLOR_GRAY)
        self.left_frame.place(relx=0, rely=0, relwidth=0.7, relheight=1.0)
        
        tk.Label(self.left_frame, text="LIVE CAMERA FEED", font=("Arial", 16, "bold"), 
                 bg=COLOR_NAVY, fg=COLOR_WHITE, pady=10).pack(fill=tk.X)
        self.video_label = tk.Label(self.left_frame, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=15, pady=15)
        
        # 2. KHU VỰC BẢNG ĐIỀU KHIỂN (30%)
        self.right_frame = tk.Frame(self.root, bg=COLOR_WHITE)
        self.right_frame.place(relx=0.7, rely=0, relwidth=0.3, relheight=1.0)
        
        tk.Label(self.right_frame, text="ĐIỀU KHIỂN BĂNG TẢI", font=("Arial", 14, "bold"), 
                 bg=COLOR_NAVY, fg=COLOR_WHITE, pady=10).pack(fill=tk.X)
        
        btn_frame = tk.Frame(self.right_frame, bg=COLOR_WHITE)
        btn_frame.pack(pady=15, fill=tk.X, padx=10)
        self.btn_start = tk.Button(btn_frame, text="START", font=("Arial", 12, "bold"), bg="#28A745", fg=COLOR_WHITE, command=self.start_conveyor)
        self.btn_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.btn_stop = tk.Button(btn_frame, text="STOP", font=("Arial", 12, "bold"), bg="#DC3545", fg=COLOR_WHITE, command=self.stop_conveyor)
        self.btn_stop.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5)
        
        tk.Label(self.right_frame, text="THỐNG KÊ SẢN LƯỢNG", font=("Arial", 14, "bold"), bg=COLOR_NAVY, fg=COLOR_WHITE, pady=10).pack(fill=tk.X, pady=(20,0))
        stat_frame = tk.Frame(self.right_frame, bg=COLOR_WHITE)
        stat_frame.pack(pady=15, fill=tk.X, padx=20)
        self.lbl_t1 = self.create_stat_box(stat_frame, "LOẠI 1 (Đạt chuẩn)", "#28A745")
        self.lbl_t2 = self.create_stat_box(stat_frame, "LOẠI 2 (Lỗi nhẹ)", "#FFC107")
        self.lbl_t3 = self.create_stat_box(stat_frame, "LOẠI 3 (Phế phẩm)", "#DC3545")
        
        tk.Label(self.right_frame, text="TRẠNG THÁI HỆ THỐNG", font=("Arial", 14, "bold"), bg=COLOR_NAVY, fg=COLOR_WHITE, pady=10).pack(fill=tk.X, pady=(20,0))
        status_frame = tk.Frame(self.right_frame, bg=COLOR_WHITE)
        status_frame.pack(pady=10, fill=tk.X, padx=20)
        self.lbl_cam = tk.Label(status_frame, text="Camera: Đang chờ...", font=("Arial", 11), bg=COLOR_WHITE, anchor="w")
        self.lbl_cam.pack(fill=tk.X)
        self.lbl_ard = tk.Label(status_frame, text="Arduino: Đang chờ...", font=("Arial", 11), bg=COLOR_WHITE, anchor="w")
        self.lbl_ard.pack(fill=tk.X)
        self.btn_cam_set = tk.Button(status_frame, text="⚙ CÀI ĐẶT ÁNH SÁNG CAMERA", font=("Arial", 10, "bold"), 
                                     bg="#17A2B8", fg=COLOR_WHITE, relief="flat", command=self.open_cam_settings)
        self.btn_cam_set.pack(fill=tk.X, pady=10)
        # -----------------------------

        self.lbl_alert = tk.Label(self.right_frame, text="", font=("Arial", 11, "bold"), bg=COLOR_WHITE)

        
        self.lbl_alert = tk.Label(self.right_frame, text="", font=("Arial", 11, "bold"), bg=COLOR_WHITE)
        self.lbl_alert.pack(side=tk.BOTTOM, pady=20)

    def create_stat_box(self, parent, title, color):
        frame = tk.Frame(parent, bg=color)
        frame.pack(fill=tk.X, pady=5)
        tk.Label(frame, text=title, font=("Arial", 11, "bold"), bg=color, fg=COLOR_WHITE).pack(side=tk.LEFT, padx=10, pady=8)
        lbl = tk.Label(frame, text="0", font=("Arial", 14, "bold"), bg=color, fg=COLOR_WHITE)
        lbl.pack(side=tk.RIGHT, padx=10)
        return lbl

    def start_conveyor(self):
        if self.link.write("R"):
            self.lbl_alert.config(text="Đã gửi lệnh chạy băng tải", fg="#28A745")
    def open_cam_settings(self):
        # Gọi lệnh mở cửa sổ Properties của OpenCV
        try:
            self.system.src.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        except Exception as e:
            self.lbl_alert.config(text="Không thể mở cài đặt camera", fg="#DC3545")

    def stop_conveyor(self):
        if self.link.write("r"):
            self.lbl_alert.config(text="Đã dừng băng tải", fg="#DC3545")

    def update_frame(self):
        # 1. RÚT DỮ LIỆU TỪ LÕI CHÍNH
        try:
            vis, dbg = self.system.disp_q.get(timeout=0.01)
            
            # Cập nhật số liệu thực tế từ rec (Recorder)
            current_count = len(self.rec.results)
            if current_count > self.last_result_count:
                # Có quả táo MỚI vừa được phân loại xong
                for i in range(self.last_result_count, current_count):
                    cls = self.rec.results[i].get('pred', 0)
                    if cls == 1: self.count_1 += 1
                    elif cls == 2: self.count_2 += 1
                    elif cls == 3: self.count_3 += 1
                
                # Cập nhật số liệu mới lên GUI
                self.lbl_t1.config(text=str(self.count_1))
                self.lbl_t2.config(text=str(self.count_2))
                self.lbl_t3.config(text=str(self.count_3))
                
                # Chốt lại số lượng đã xử lý để không đếm trùng ở frame sau
                self.last_result_count = current_count

            # Dán hình lên giao diện Tkinter
            frame_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((850, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            self.lbl_cam.config(text="Camera: Đang kết nối (Tốt)", fg="#28A745")
        except queue.Empty:
            pass

        # 2. Cập nhật trạng thái Hardware
        status = self.gates.status()
        if status.get('link_ok'):
            self.lbl_ard.config(text=f"Arduino: {status.get('name')} (Sẵn sàng)", fg="#28A745")
        else:
            self.lbl_ard.config(text="Arduino: Mất kết nối!", fg="#DC3545")

        self.root.after(30, self.update_frame)

    def on_closing(self):
        self.system.stop()
        self.gates.safe()
        self.link.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppleSorterGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
