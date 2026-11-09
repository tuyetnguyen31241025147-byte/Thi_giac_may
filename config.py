"""
config.py - TOAN BO THONG SO CUA HE THONG 
Toa do tren bang tai tinh bang cm, goc 0 = dau bang tai (noi tha tao).
"""
# ============ CAMERA (Lens + Sensor + Interface) ============
CAM_INDEX = 1
CAM_WIDTH, CAM_HEIGHT = 1280, 720      # webcam 1080P chay 720p la du nhanh va nhe hon
LOCK_CAMERA_AUTO = True                # co gang khoa tu phoi sang / lay net / can bang trang
CAM_TIMEOUT_S = 2.0                    # qua thoi gian nay khong co frame -> coi la MAT CAMERA

# ============ ARDUINO (Interface -> Controller) ============
SERIAL_PORT = "COM10"
SERIAL_BAUD = 9600
SERIAL_RECONNECT_S = 2.0

# ============ HINH HOC BANG TAI (cm) ============
BELT_LEN_CM = 40.0
ROI_X0_CM = 7.5                        # mep TRAI cua vung quan sat (ROI) nam o 7,5 cm
ROI_LEN_CM = 11.0                      # ROI dai 11 cm (7,5 -> 18,5)
GATE_PIVOT_CM = {1: 24.0, 2: 32.0}     # truc quay can gat 1 va 2
GATE_LEN_CM = 7.0                      # chieu dai can gat
BELT_SPEED_MM_S = 47.0                 # toc do bang tai SAU buong camera (DO bang dong ho bam gio)
GATE_OFFSET_S = {1: 0.0, 2: 0.0}       # tinh chinh thoi diem theo thuc nghiem (+ = tre hon)

OPEN_LEAD_S = 0.8                      # mo can SOM hon luc qua toi (servo + serial + du phong)

CLOSE_MARGIN_CM = 3.0                  # dong can khi duoi qua cach dau can >= 1 cm
MERGE_GAP_S = 0.6                      # hai lan mo cach nhau < 0,6 s thi gop (do servo giat)

# ============ PHAN LOAI ============
AREA_THRESHOLD_MM2 = 1590.0            # >= : Loai 1 (d >= 45 mm), < : Loai 2
KNOWN_D_MM = 45.0                      # duong kinh vat mau dung hieu chuan ty le mm/pixel

# ============ XU LY ANH ============
BG_METHOD = "lab"                      # "lab" (khoang cach mau Lab) hoac "gray" (hieu muc xam)
FG_THRESH = 25                         # nguong tach tao khoi nen
MEDIAN_MM = 1.2                        # kich thuoc loc trung vi (mm)
MORPH_MM = 2.0                         # kich thuoc phan tu hinh thai hoc (mm)
MIN_BLOB_MM2 = 300.0                   # vat the nho hon -> nhieu
GREEN_LO, GREEN_HI = (25, 40, 40), (90, 255, 255)        # HSV cua vo tao xanh tuoi
HIGHLIGHT_LO, HIGHLIGHT_HI = (0, 0, 220), (179, 50, 255) # vet loa sang (bo qua)
CANNY_LO, CANNY_HI = 40, 110
ERODE_MM = 2.5                         # co mat na vao de bo vien toi o mep qua
MIN_DEFECT_MM2 = 5.0                   # contour loi nho hon -> nhieu
# --- phat hien khuyet tat: 3 "tin hieu" gop lai (xem vision.py) ---
#  (1) LECH MAU SO VOI DA TAO CUA CHINH QUA (nhay voi vet dap nhat/vua; khong dung dai xanh co dinh)
DEFECT_DE_MIN = 10.0                   # san do lech mau Lab toi thieu (don vi OpenCV; 2.55 don vi ~ 1 L*)
DEFECT_K = 3.5                         # nguong = trung vi + K x do phan tan (MAD) cua chinh da tao
#  (2) khong thuoc dai xanh HSV (vet nau/den dam)   (3) net Canny trong long qua (nut, xuoc)
SKIN_POLY_DEG = 3                      # bac da thuc mo ta bong do cua da qua (2-4); cao hon = it nhay vet to

DEFECT_RATIO = 0.05                    # tong dien tich loi / dien tich qua > 1% -> frame bao loi
DEFECT_BLOB_MM2 = 1e9                 # HOAC co 1 vung loi >= 15 mm2 -> frame bao loi
STRONG_RATIO = 1.0                 # 1 frame loi > 8% dien tich -> Loai 3 ngay
STRONG_BLOB_MM2 = 1e9               # HOAC 1 vung loi >= 80 mm2 -> Loai 3 ngay

# ============ GOP NHIEU FRAME (qua tao xoay) ============
DECISION_X_FRAC = 0.85                 # quyet dinh khi tam qua vuot 85% chieu dai ROI
MATCH_MM = 25.0                        # ghep qua giua 2 frame neu tam lech < 25 mm
LOST_FRAMES = 8
MIN_FRAMES = 5                         # it hon -> canh bao do tin cay thap
MIN_DEFECT_FRAMES = 2                  # >= 2 frame bao loi -> Loai 3

# ============ GIAM SAT TINH HUONG ============
LIGHT_TOL = 0.15                       # do sang nen lech > 15% so voi luc chup nen -> canh bao
FOCUS_TOL = 0.5                        # do net giam duoi 50% so voi luc chup nen -> canh bao
