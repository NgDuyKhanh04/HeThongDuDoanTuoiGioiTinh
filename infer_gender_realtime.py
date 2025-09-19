
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =======================
# 1) THAM SO CAU HINH
# =======================
MO_HINH = "age_gender_model_v3.keras"  
KICH_THUOC_ANH = 224                   # Resize patch khuon mat
KHUON_MAT_TOI_THIEU = 40               # Bo qua mat qua nho
DNN_NGUONG = 0.45                      # Nguong tin cay DNN
PHONG_TO = 1.3                         # He so phong to khung khi detect (can bang toc do)

NGUONG_GIOI_TINH = 0.50                # p<0.5 -> Nam; p>0.5 -> Nu
BIEN_DO_KHONG_RO = 0.10                # |p-0.5| <= 0.1 -> "Khong ro"

NHAN_NAM = "Nam"
NHAN_NU = "Nu"
NHAN_KHONG_RO = "Khong ro"

# ==============================================
# 2) CHON BO DO KHUON MAT
# ==============================================
USE_DNN = os.path.exists("deploy.prototxt") and os.path.exists("res10_300x300_ssd_iter_140000.caffemodel")

if USE_DNN:
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    def phat_hien_khuon_mat(bgr, conf=DNN_NGUONG):
        """
        Tra ve list [(x, y, w, h), ...] khuon mat.
        Phong to khung hinh truoc khi detect de bat mat nho -> quy ve toa do goc.
        """
        lon = cv2.resize(bgr, None, fx=PHONG_TO, fy=PHONG_TO, interpolation=cv2.INTER_LINEAR)
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
                # Quy doi ve anh goc
                x1 = int(x1 / PHONG_TO); y1 = int(y1 / PHONG_TO)
                x2 = int(x2 / PHONG_TO); y2 = int(y2 / PHONG_TO)
                if (x2 - x1) >= KHUON_MAT_TOI_THIEU and (y2 - y1) >= KHUON_MAT_TOI_THIEU:
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
else:
    FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def phat_hien_khuon_mat(bgr, conf=0.6):
        """
        Tra ve list [(x, y, w, h), ...] bang Haar Cascade.
        Tang nhay cho mat nho bang scaleFactor thap va minNeighbors thap hon.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(KHUON_MAT_TOI_THIEU, KHUON_MAT_TOI_THIEU),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return list(faces) if faces is not None else []

# ==================================
# 3) TIEN XU LY PATCH CHO MODEL
# ==================================
def chuan_hoa_patch(bgr):
    """
    Resize -> RGB -> float32 -> them batch dimension.
    
    """
    x = cv2.resize(bgr, (KICH_THUOC_ANH, KICH_THUOC_ANH))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype("float32")
    return np.expand_dims(x, 0)

# ==================================================
# 4) VE NHAN TRANH CHONG (don gian, nhanh)
# ==================================================
VUNG_NHAN_DA_VE = []  # danh sach (x1,y1,x2,y2) cho frame hien tai

def _hop_giao(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def _kep(x1, y1, w, h, W, H):
    x1 = max(0, min(x1, W - w))
    y1 = max(0, min(y1, H - h))
    return x1, y1, x1 + w, y1 + h

def ve_nhan(img, x, y, w, h, text):
    """Ve khung mat + nhan, co tranh chong nhan voi nhau."""
    # Khung mat
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Kich thuoc nhan
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 6
    bw, bh = tw + pad + 3, th + pad
    H, W = img.shape[:2]

    # Cac vi tri ung vien
    candidates = [
        (x,          y - 4 - bh),   # tren
        (x,          y + h + 4),    # duoi
        (x + w//2,   y - 4 - bh),   # tren le phai
        (x - w//2,   y - 4 - bh),   # tren le trai
        (x + w//2,   y + h + 4),    # duoi le phai
        (x + 2,      y + 2),        # ben trong (fallback)
    ]

    chosen = None
    for (bx1, by1) in candidates:
        cx1, cy1, cx2, cy2 = _kep(bx1, by1, bw, bh, W, H)
        if not any(_hop_giao((cx1, cy1, cx2, cy2), area) for area in VUNG_NHAN_DA_VE):
            chosen = (cx1, cy1, cx2, cy2)
            break
    if chosen is None:
        cx1, cy1, cx2, cy2 = _kep(x + 2, y + 2, bw, bh, W, H)
        chosen = (cx1, cy1, cx2, cy2)

    cx1, cy1, cx2, cy2 = chosen
    cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (0, 0, 0), -1)
    cv2.putText(img, text, (cx1 + 3, cy2 - pad // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    VUNG_NHAN_DA_VE.append(chosen)

# =======================
# 5) CHUONG TRINH CHINH
# =======================
def main():
    # Load mo hinh (2 dau ra) nhung CHI dung output gioi tinh
    model = load_model(MO_HINH, compile=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Khong mo duoc webcam"); 
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Clear ds nhan cho tung frame
        VUNG_NHAN_DA_VE.clear()

        # 1) Phat hien mat
        boxes = phat_hien_khuon_mat(frame)

        # 2) Du doan gioi tinh tung mat
        for (x, y, w, h) in boxes:
            roi = frame[y:y + h, x:x + w]
            xin = chuan_hoa_patch(roi)

            # model.predict tra ve [y_age, y_gender]; bo y_age
            _, y_gender = model.predict(xin, verbose=0)
            p = float(y_gender[0][0])

            if abs(p - NGUONG_GIOI_TINH) <= BIEN_DO_KHONG_RO:
                nhan = NHAN_KHONG_RO
            else:
                nhan = NHAN_NAM if p < NGUONG_GIOI_TINH else NHAN_NU

            ve_nhan(frame, x, y, w, h, nhan)

        # 3) Hien thi
        cv2.imshow("Realtime - Gioi tinh ", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Diem vao
if __name__ == "__main__":
    main()
