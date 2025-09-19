
import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =======================
# 1) THAM SỐ CẤU HÌNH
# =======================
MO_HINH = "age_gender_model_v3.keras" 
KICH_THUOC_ANH = 224                   # Resize patch khuôn mặt về 224x224
KHUON_MAT_TOI_THIEU = 40               # Bỏ qua mặt quá nhỏ (<40px)
DNN_NGUONG = 0.45                      # Confidence threshold cho DNN
PHONG_TO = 1.5                         # Hệ số phóng to ảnh để bắt mặt nhỏ
NGUONG_GIOI_TINH = 0.50                # p<0.5 -> Nam; p>0.5 -> Nu
BIEN_DO_KHONG_RO = 0.10                # |p-0.5| <= 0.1 -> "Khong ro"

NHAN_NAM = "Nam"
NHAN_NU = "Nu"
NHAN_KHONG_RO = "Khong ro"

# ==============================================
# 2) CHỌN BỘ DÒ KHUÔN MẶT
# ==============================================
USE_DNN = os.path.exists("deploy.prototxt") and os.path.exists("res10_300x300_ssd_iter_140000.caffemodel")

if USE_DNN:
    # DNN SSD ResNet10 (chính xác & ổn định hơn Haar cho ảnh nhóm)
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    def phat_hien_khuon_mat(bgr, conf=DNN_NGUONG):
        """
        Trả về list [(x, y, w, h), ...] các bounding box khuôn mặt trên ảnh bgr.
        - Ảnh được phóng to trước khi detect để tăng tỉ lệ mặt nhỏ.
        - Sau khi detect, toạ độ được quy đổi về hệ của ảnh gốc.
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
                # Quy đổi về toạ độ ảnh gốc
                x1 = int(x1 / PHONG_TO); y1 = int(y1 / PHONG_TO)
                x2 = int(x2 / PHONG_TO); y2 = int(y2 / PHONG_TO)
                if (x2 - x1) >= KHUON_MAT_TOI_THIEU and (y2 - y1) >= KHUON_MAT_TOI_THIEU:
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
else:
    # Haar Cascade (nhanh, nhưng yếu hơn DNN với ảnh nhóm/xa/nghiêng)
    FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def phat_hien_khuon_mat(bgr, conf=0.6):
        """
        Trả về list [(x, y, w, h), ...] khuôn mặt bằng Haar.
        Tăng nhạy cho mặt nhỏ bằng scaleFactor thấp và minNeighbors nhỏ hơn.
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
# 3) TIỀN XỬ LÝ PATCH KHUÔN MẶT
# ==================================
def doc_anh_chuan_hoa(bgr):
    """
    - Cắt/resize patch khuôn mặt về (224, 224), chuyển BGR -> RGB, float32.
    - Lưu ý: model v3 đã xử lý chuẩn hoá trong pipeline training;
             ở infer ta giữ float32, KHÔNG chia 255 ở đây.
    """
    x = cv2.resize(bgr, (KICH_THUOC_ANH, KICH_THUOC_ANH))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype("float32")
    return np.expand_dims(x, 0)  # (1, H, W, 3)

# ==================================================
# 4) VẼ NHÃN TRÁNH CHỒNG: THỬ NHIỀU VỊ TRÍ AN TOÀN
# ==================================================
VUNG_NHAN_DA_VE = []  # các hộp nhãn đã vẽ: (x1, y1, x2, y2)

def _hop_giao_nhau(a, b):
    """Kiểm tra 2 hình chữ nhật có giao nhau không."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def _kep_trong_khung(x1, y1, w, h, W, H):
    """Kẹp 1 hộp (x1,y1,w,h) nằm hoàn toàn trong khung ảnh W x H."""
    x1 = max(0, min(x1, W - w))
    y1 = max(0, min(y1, H - h))
    return x1, y1, x1 + w, y1 + h

def ve_nhan_tu_dong_tranh_chong(img, x, y, w, h, text):
    """Vẽ nhãn tại vị trí 'hợp lý' nhất để không chồng nhãn trước đó."""
    # Vẽ khung mặt
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Tính kích thước nhãn
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 6
    box_w, box_h = tw + pad + 3, th + pad
    H, W = img.shape[:2]

    # Danh sách vị trí ứng viên (ưu tiên theo thứ tự)
    candidates = [
        (x,           y - 4 - box_h),       # trên
        (x,           y + h + 4),           # dưới
        (x + w // 2,  y - 4 - box_h),       # trên lệch phải
        (x - w // 2,  y - 4 - box_h),       # trên lệch trái
        (x + w // 2,  y + h + 4),           # dưới lệch phải
        (x + 2,       y + 2),               # trong khung (fallback)
    ]

    chon = None
    for (bx1, by1) in candidates:
        cx1, cy1, cx2, cy2 = _kep_trong_khung(bx1, by1, box_w, box_h, W, H)
        if not any(_hop_giao_nhau((cx1, cy1, cx2, cy2), area) for area in VUNG_NHAN_DA_VE):
            chon = (cx1, cy1, cx2, cy2)
            break

    # Nếu tất cả đều chồng -> buộc vẽ trong khung
    if chon is None:
        cx1, cy1, cx2, cy2 = _kep_trong_khung(x + 2, y + 2, box_w, box_h, W, H)
        chon = (cx1, cy1, cx2, cy2)

    cx1, cy1, cx2, cy2 = chon
    cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (0, 0, 0), -1)
    cv2.putText(img, text, (cx1 + 3, cy2 - pad // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    VUNG_NHAN_DA_VE.append(chon)

# =======================
# 5) CHƯƠNG TRÌNH CHÍNH
# =======================
def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python infer_gender_image.py <duong_dan_anh>")
        return

    duong_dan = sys.argv[1]
    if not os.path.exists(duong_dan):
        print(" Không tìm thấy file:", duong_dan)
        return

    img = cv2.imread(duong_dan)
    if img is None:
        print(" Không đọc được ảnh (định dạng không hỗ trợ hoặc file lỗi).")
        return

    # Load mô hình 2 đầu ra, nhưng CHỈ dùng output thứ 2 (gioitinh)
    model = load_model(MO_HINH, compile=False)

    # B1: Phát hiện khuôn mặt
    boxes = phat_hien_khuon_mat(img)
    if not boxes:
        print(" Không phát hiện được khuôn mặt. Thử giảm KHUON_MAT_TOI_THIEU hoặc tăng PHONG_TO.")
    # Reset danh sách vùng nhãn cho ảnh này
    VUNG_NHAN_DA_VE.clear()

    # B2: Dự đoán giới tính cho từng mặt
    for (x, y, w, h) in boxes:
        patch = img[y:y + h, x:x + w]
        xin = doc_anh_chuan_hoa(patch)

        # model.predict trả về [y_age, y_gender]; ta bỏ y_age
        _, y_gender = model.predict(xin, verbose=0)
        p = float(y_gender[0][0])  # sigmoid ∈ (0,1)

        if abs(p - NGUONG_GIOI_TINH) <= BIEN_DO_KHONG_RO:
            nhan = NHAN_KHONG_RO
        else:
            nhan = NHAN_NAM if p < NGUONG_GIOI_TINH else NHAN_NU

        ve_nhan_tu_dong_tranh_chong(img, x, y, w, h, nhan)

    # B3: Hiển thị kết quả
    cv2.imshow("Gender Only", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Điểm vào chương trình
if __name__ == "__main__":
    main()
