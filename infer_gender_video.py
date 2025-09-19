
import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =======================
# 1) THAM SO CAU HINH
# =======================
MO_HINH = "age_gender_model_v3.keras"   
KICH_THUOC_ANH = 224                    # resize patch khuon mat
KHUON_MAT_TOI_THIEU = 40                # bo qua mat < 40px
DNN_NGUONG = 0.45                       # nguong tin cay DNN (thap de bat mat nho/xa)
PHONG_TO_DETECT = 1.5                   # phong to frame khi DETECT (khong anh huong xuat/hiển thị)

NGUONG_GIOI_TINH = 0.50                 # p < 0.5 -> Nam ; p > 0.5 -> Nu
BIEN_DO_KHONG_RO = 0.10                 # |p-0.5| <= 0.1 -> "Khong ro"

NHAN_NAM = "Nam"
NHAN_NU = "Nu"
NHAN_KHONG_RO = "Khong ro"

# Hiển thị: thu nhỏ cửa sổ "
WINDOW_NAME = "Video - Gioi tinh "
DISPLAY_SCALE = 0.6   # 60% size goc khi HIEN THI (file xuat van giu size goc)

# ==============================================
# 2) CHON BO DO KHUON MAT: DNN neu co / HAAR
# ==============================================
USE_DNN = os.path.exists("deploy.prototxt") and os.path.exists("res10_300x300_ssd_iter_140000.caffemodel")

if USE_DNN:
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    def phat_hien_khuon_mat(bgr, conf=DNN_NGUONG):
        """
        Tra ve list [(x, y, w, h), ...] bbox khuon mat.
        - Phong to frame truoc khi detect -> bat mat nho.
        - Sau do QUY DOI toa do ve frame goc.
        """
        lon = cv2.resize(bgr, None, fx=PHONG_TO_DETECT, fy=PHONG_TO_DETECT, interpolation=cv2.INTER_LINEAR)
        H, W = lon.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(lon, (300, 300)),
                                     scalefactor=1.0, size=(300, 300),
                                     mean=(104, 177, 123))
        net.setInput(blob)
        det = net.forward()
        boxes = []
        for i in range(det.shape[2]):
            c = float(det[0, 0, i, 2])
            if c >= conf:
                x1, y1, x2, y2 = det[0, 0, i, 3:7] * [W, H, W, H]
                x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)])
                # quy doi ve frame goc
                x1 = int(x1 / PHONG_TO_DETECT); y1 = int(y1 / PHONG_TO_DETECT)
                x2 = int(x2 / PHONG_TO_DETECT); y2 = int(y2 / PHONG_TO_DETECT)
                w = x2 - x1; h = y2 - y1
                if w >= KHUON_MAT_TOI_THIEU and h >= KHUON_MAT_TOI_THIEU:
                    boxes.append((x1, y1, w, h))
        return boxes
else:
    FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def phat_hien_khuon_mat(bgr, conf=0.6):
        """Detect khuon mat bang Haar (nhanh nhung yeu hon DNN)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3,
            minSize=(KHUON_MAT_TOI_THIEU, KHUON_MAT_TOI_THIEU),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return list(faces) if faces is not None else []

# ==================================
# 3) TIEN XU LY PATCH CHO MODEL
# ==================================
def chuan_hoa_patch(bgr):
    """
    - Cat patch khuon mat -> resize 224x224
    - BGR -> RGB -> float32 -> them batch dimension
    Luu y: mo hinh v3 da co preprocess trong pipeline train; giu float32 (khong chia 255 tai day).
    """
    x = cv2.resize(bgr, (KICH_THUOC_ANH, KICH_THUOC_ANH))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype("float32")
    return np.expand_dims(x, 0)  # (1, 224, 224, 3)

# ==================================
# 4) VE NHAN (dat TRONG khung de khong de nhau)
# ==================================
def ve_nhan_trong_khung(img, x, y, w, h, text):
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 6
    bx1, by1 = x + 2, y + 2
    cv2.rectangle(img, (bx1, by1), (bx1 + tw + 8, by1 + th + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (bx1 + 3, by1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# =======================
# 5) CHUONG TRINH CHINH
# =======================
def main():
    # ---- Kiem tra tham so ----
    if len(sys.argv) < 2:
        print("Cach dung: python infer_gender_video.py <duong_dan_video>")
        return
    duong_dan_video = sys.argv[1]
    if not os.path.exists(duong_dan_video):
        print(" Khong thay file:", duong_dan_video); return

    # ---- Load mo hinh (2 dau ra) nhung CHI dung dau ra gioi tinh ----
    model = load_model(MO_HINH, compile=False)

    # ---- Mo video ----
    cap = cv2.VideoCapture(duong_dan_video)
    if not cap.isOpened():
        print(" Khong mo duoc video"); return

    # ---- Thong so khung + writer ----
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25
    out_path = os.path.splitext(duong_dan_video)[0] + "_gender.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    # ---- Cua so hien thi (nho lai) ----
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, int(W * DISPLAY_SCALE), int(H * DISPLAY_SCALE))

    # ---- Xu ly tung frame ----
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Detect khuon mat
        boxes = phat_hien_khuon_mat(frame)

        # 2) Du doan gioi tinh tung khuon mat
        for (x, y, w, h) in boxes:
            patch = frame[y:y + h, x:x + w]
            xin = chuan_hoa_patch(patch)

            # model.predict -> [y_age, y_gender] ; bo y_age
            _, y_gender = model.predict(xin, verbose=0)
            p = float(y_gender[0][0])  # sigmoid 0..1

            if abs(p - NGUONG_GIOI_TINH) <= BIEN_DO_KHONG_RO:
                nhan = NHAN_KHONG_RO
            else:
                nhan = NHAN_NAM if p < NGUONG_GIOI_TINH else NHAN_NU

            ve_nhan_trong_khung(frame, x, y, w, h, nhan)

        # 3) Ghi file goc + hien thi thu nho
        writer.write(frame)
        hien_thi = frame if DISPLAY_SCALE == 1.0 else cv2.resize(
            frame, (int(W * DISPLAY_SCALE), int(H * DISPLAY_SCALE)), interpolation=cv2.INTER_AREA
        )
        cv2.imshow(WINDOW_NAME, hien_thi)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    # ---- Giai phong ----
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(" Da xuat video:", out_path)

# Diem vao
if __name__ == "__main__":
    main()
