
# Dự đoán TUỔI + GIỚI TÍNH trên 1 ảnh
# ==============================

import sys, os, cv2, numpy as np
from tensorflow.keras.models import load_model

# ---- Cấu hình ----
DUONG_DAN_MO_HINH = "age_gender_model_1.keras"
THU_MUC_MAC_DINH = os.path.join("data", "utkface")  
KICH_THUOC_ANH = 64
TUOI_TOI_DA = 100.0
MIN_FACE_SIZE = 80  # px

# Nhãn hiển thị
TEXT_MALE = "Nam"
TEXT_FEMALE = "Nu"
TEXT_UNKNOWN = "Khong ro"

# Ngưỡng quyết định
NGUONG_GIOITINH = 0.50    # < ngưỡng => Nam, >= ngưỡng => Nữ
BIEN_DO_GIOITINH = 0.10   # vùng mơ hồ quanh ngưỡng -> "Không rõ"
TIN_CAY_NHOM = 0.40       # tự tin nhóm tuổi (softmax) dưới mức này 

NHOM_TUOI_TEN  = ["0-12","13-17","18-24","25-34","35-44","45-54","55-64","65+"]

def _lam_dep_duong_dan(p: str) -> str:
    """Bỏ nháy + khoảng trắng + mở rộng ~"""
    if not p: return p
    p = p.strip().strip('"').strip("'").strip()
    return os.path.expanduser(p)

def _ghep_thu_muc_mac_dinh_neu_can(p: str) -> str:
    """
    Nếu người dùng CHỈ nhập tên file (không có thư mục) và file không tồn tại,
    tự động ghép THU_MUC_MAC_DINH/tên_file.
    """
    if os.path.exists(p):
        return p
    if os.path.dirname(p) == "":  # chỉ tên file
        p2 = os.path.join(THU_MUC_MAC_DINH, p)
        if os.path.exists(p2):
            return p2
    return p  # giữ nguyên nếu vẫn không thấy (để báo lỗi)

def _tien_xu_ly(face_bgr):
    im = cv2.resize(face_bgr, (KICH_THUOC_ANH, KICH_THUOC_ANH))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype("float32")/255.0
    return np.expand_dims(im, 0)

def _ve_khung(frame, x,y,w,h, text):
    # vẽ khung + nền chữ cho dễ đọc
    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y-8-th-6), (x+tw+6, y-4), (0,0,0), -1)
    cv2.putText(frame, text, (x+3, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def main():
    # --- Lấy đường dẫn hoặc tên file ---
    if len(sys.argv) >= 2:
        img_path = " ".join(sys.argv[1:])  # gộp nếu có khoảng trắng
    else:
        print("python infer_image.py <duong_dan_anh> (hoặc chỉ nhập tên file trong data/utkface)")
        try:
            img_path = input("Nhập đường dẫn ảnh hoặc TÊN FILE (trong data/utkface): ")
        except EOFError:
            img_path = ""

    img_path = _lam_dep_duong_dan(img_path)
    img_path = _ghep_thu_muc_mac_dinh_neu_can(img_path)

    if not os.path.exists(img_path):
        print(" Không thấy file:", img_path)
        return

    # Load model (không compile để khỏi cần custom metric)
    try:
        model = load_model(DUONG_DAN_MO_HINH, compile=False)
    except Exception as e:
        print(" Lỗi load mô hình:", e)
        return

    img = cv2.imread(img_path)
    if img is None:
        print(" Không đọc được ảnh. Kiểm tra đường dẫn/định dạng.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))

    if faces is None or len(faces) == 0:
        print(" Không phát hiện khuôn mặt.")
        out_path = os.path.splitext(img_path)[0] + "_out.jpg"
        cv2.imwrite(out_path, img); print("Đã lưu:", out_path)
        return

    for (x,y,w,h) in faces:
        roi = img[y:y+h, x:x+w]
        if roi.size == 0: 
            continue
        x_in = _tien_xu_ly(roi)
        y_pred = model.predict(x_in, verbose=0)
        if not isinstance(y_pred, (list, tuple)) or len(y_pred) != 3:
            print(" Model không có 3 đầu ra."); continue

        p_nhom, p_tuoi, p_gioi = y_pred
        # Nhóm tuổi
        idx = int(np.argmax(p_nhom[0]))
        conf = float(p_nhom[0][idx])
        scope = NHOM_TUOI_TEN[idx] if conf >= TIN_CAY_NHOM else f"{NHOM_TUOI_TEN[idx]}?"
        # Tuổi (0..1)*100
        age = int(round(float(p_tuoi[0][0]) * TUOI_TOI_DA))
        # Giới tính
        prob = float(p_gioi[0][0])  # 0..1
        if abs(prob - NGUONG_GIOITINH) <= BIEN_DO_GIOITINH:
            g = TEXT_UNKNOWN
        else:
            g = TEXT_MALE if prob < NGUONG_GIOITINH else TEXT_FEMALE

        _ve_khung(img, x,y,w,h, f"{g}, {scope} (~{age})")
        print(f"[{os.path.basename(img_path)} | ({x},{y},{w},{h})] -> {g}, {scope}, ~{age} (conf_nhom={conf:.2f}, p_gioi={prob:.2f})")

    out_path = os.path.splitext(img_path)[0] + "_out.jpg"
    cv2.imwrite(out_path, img)
    print(" Đã lưu:", out_path)
    cv2.imshow("Ket qua du doan", img); cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
