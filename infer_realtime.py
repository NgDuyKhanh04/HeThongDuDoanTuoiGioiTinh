
# - Mô hình có 3 đầu ra theo thứ tự:
#   [tuoi_nhom (softmax 8 lớp), tuoi (sigmoid 0..1), gioitinh (sigmoid 0..1)]
# PHÍM TẮT:
# - 'q' : thoát chương trình
# - 'a' : popup tóm tắt các khuôn mặt đang thấy
# ==============================

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import OrderedDict
import tkinter as tk
from tkinter import messagebox

# ------------------------------
# Cấu hình chung
# ------------------------------
DUONG_DAN_MO_HINH = "age_gender_model_1.keras"  
KICH_THUOC_ANH = 64
TUOI_TOI_DA = 100.0

# Camera index:
# - Webcam laptop: 0
# - Webcam USB thứ 2: 1, 2, ...
# - iPhone qua app IP camera: đặt URL, vd: "http://192.168.1.50:8080/video" hoặc "rtsp://..."
CAM_INDEX = 0

# Nhóm tuổi (scope) để hiển thị
NHOM_TUOI_BIEN = [(0,12),(13,17),(18,24),(25,34),(35,44),(45,54),(55,64),(65,150)]
NHOM_TUOI_TEN  = ["0-12","13-17","18-24","25-34","35-44","45-54","55-64","65+"]

# Làm mượt (EMA)
ALPHA_TUOI = 0.6
ALPHA_GIOITINH = 0.6
ALPHA_NHOM = 0.6

# Ngưỡng quyết định / tự tin
NGUONG_GIOITINH = 0.50   # ngưỡng quyết định nam/nữ
BIEN_DO_GIOITINH = 0.10  # vùng mơ hồ quanh ngưỡng (0.40..0.60 => Không rõ)
TIN_CAY_NHOM = 0.40      # hạ từ 0.60 xuống 0.40 để hạn chế "??" (có thể giảm tiếp)

# Bỏ qua mặt quá nhỏ (giảm sai do ảnh mờ/xa)
MIN_FACE_SIZE = 80  # px

# ------------------------------
# Hàm tiện ích
# ------------------------------
def tien_xu_ly_khuon_mat(face_bgr):
    """Chuẩn hóa ảnh khuôn mặt về (1, KICH_THUOC_ANH, KICH_THUOC_ANH, 3) theo RGB và /255."""
    face = cv2.resize(face_bgr, (KICH_THUOC_ANH, KICH_THUOC_ANH))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

def lam_muot_ema(gia_tri_cu, gia_tri_moi, alpha):
    """Exponential Moving Average để làm mượt theo thời gian."""
    if gia_tri_cu is None:
        return gia_tri_moi
    return gia_tri_cu * (1 - alpha) + gia_tri_moi * alpha

def gan_id_khuon_mat(faces, tracks, nguong=110):
    """
    Gán ID tạm cho bbox mặt theo vị trí gần nhất.
    faces: list[(x,y,w,h)]
    tracks: OrderedDict {id: {'cx','cy','bbox','p_tuoi_nhom','tuoi','gioitinh'}}
    """
    centers = [(x + w//2, y + h//2) for (x,y,w,h) in faces]
    used = set()
    new_tracks = OrderedDict()

    # Ghép id cũ với detection mới (gần nhất và trong ngưỡng)
    for tid, st in tracks.items():
        if not centers:
            continue
        dists = [abs(cx - st['cx']) + abs(cy - st['cy']) for (cx, cy) in centers]
        j = int(np.argmin(dists))
        if j in used or dists[j] > nguong:
            continue
        cx, cy = centers[j]
        new_tracks[tid] = {
            'cx': cx, 'cy': cy,
            'bbox': faces[j],
            'p_tuoi_nhom': st.get('p_tuoi_nhom'),
            'tuoi': st.get('tuoi'),
            'gioitinh': st.get('gioitinh')
        }
        used.add(j)

    # Tạo id mới cho detection chưa gán
    next_id = (max(tracks.keys()) + 1) if len(tracks) else 1
    for k, (cx, cy) in enumerate(centers):
        if k in used: 
            continue
        new_tracks[next_id] = {'cx': cx, 'cy': cy, 'bbox': faces[k],
                               'p_tuoi_nhom': None, 'tuoi': None, 'gioitinh': None}
        next_id += 1

    return new_tracks

def ve_nhan_khung(frame, x, y, w, h, text):
    """Vẽ khung và nhãn lên khung hình."""
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
    # Nền mờ cho chữ dễ đọc
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y-8-th-6), (x+tw+6, y-4), (0,0,0), -1)
    cv2.putText(frame, text, (x+3, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# ------------------------------
# Chương trình chính
# ------------------------------
def main():
    # 1) Load mô hình — dùng compile=False để khỏi cần custom metric khi load
    try:
        model = load_model(DUONG_DAN_MO_HINH, compile=False)
    except Exception as e:
        print(" Lỗi load mô hình:", e)
        print(" Kiểm tra lại DUONG_DAN_MO_HINH hoặc quyền truy cập.")
        return
    print(" Đã load mô hình:", DUONG_DAN_MO_HINH)

    # 2) Mở nguồn hình ảnh (webcam hoặc URL)
    #    Cho phép CAM_INDEX là int (0,1,...) hoặc string URL ("http://...", "rtsp://...")
    if isinstance(CAM_INDEX, str) and CAM_INDEX.isdigit():
        src = int(CAM_INDEX)
    else:
        src = CAM_INDEX
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(" Không mở được nguồn! Thử CAM_INDEX = 1 (camera phụ) hoặc dùng URL iPhone.")
        return

    # 3) Bộ dò mặt Haar Cascade (mặc định của OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print(" Không load được haarcascade_frontalface_default.xml.")
        return

    # 4) Bộ nhớ track để làm mượt theo từng mặt
    tracks = OrderedDict()  # id -> state

    while True:
        ok, frame = cap.read()
        if not ok:
            print(" Không đọc được frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_np = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )
        faces = list(faces_np) if faces_np is not None else []

        # Gán/duy trì ID cho các mặt & lưu bbox
        tracks = gan_id_khuon_mat(faces, tracks, nguong=110)

        # Lưu kết quả để popup
        bang_tom_tat = []

        # Dự đoán cho từng track có bbox
        for tid, st in tracks.items():
            bbox = st.get('bbox')
            if bbox is None:
                continue
            x, y, w, h = bbox
            # Bỏ qua nếu quá nhỏ (an toàn)
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue

            x_in = tien_xu_ly_khuon_mat(face_roi)

            try:
                # Mong đợi 3 đầu ra: [p_tuoi_nhom (1,8), p_tuoi (1,1), p_gioitinh (1,1)]
                y_pred = model.predict(x_in, verbose=0)
            except Exception as e:
                print(" Lỗi dự đoán:", e)
                continue

            if not isinstance(y_pred, (list, tuple)) or len(y_pred) != 3:
                print(" Mô hình không có 3 đầu ra (tuoi_nhom, tuoi, gioitinh). Kiểm tra lại model.")
                continue

            p_nhom, p_tuoi, p_gioitinh = y_pred

            # ===== Nhóm tuổi =====
            p_nhom_vec = p_nhom[0]  # shape (8,)
            if st['p_tuoi_nhom'] is None:
                st['p_tuoi_nhom'] = p_nhom_vec
            else:
                st['p_tuoi_nhom'] = lam_muot_ema(st['p_tuoi_nhom'], p_nhom_vec, ALPHA_NHOM)

            idx_nhom = int(np.argmax(st['p_tuoi_nhom']))
            conf_nhom = float(st['p_tuoi_nhom'][idx_nhom])
            # Nếu chưa đủ tự tin -> vẫn hiện nhóm + dấu '?' thay vì "??"
            scope = NHOM_TUOI_TEN[idx_nhom] if conf_nhom >= TIN_CAY_NHOM else f"{NHOM_TUOI_TEN[idx_nhom]}?"

            # ===== Tuổi (0..1)*100, làm mượt theo năm =====
            age_years = float(p_tuoi[0][0]) * TUOI_TOI_DA
            st['tuoi'] = lam_muot_ema(st['tuoi'], age_years, ALPHA_TUOI)
            tuoi_hienthi = int(round(st['tuoi'] if st['tuoi'] is not None else age_years))

            # ===== Giới tính (0..1), làm mượt + vùng mơ hồ =====
            gioitinh_score = float(p_gioitinh[0][0])
            st['gioitinh'] = lam_muot_ema(st['gioitinh'], gioitinh_score, ALPHA_GIOITINH)
            prob = st['gioitinh'] if st['gioitinh'] is not None else gioitinh_score  # 0..1
            if abs(prob - NGUONG_GIOITINH) <= BIEN_DO_GIOITINH:
                g_display = "Khong ro"
            else:
                g_display = "Nam" if prob < NGUONG_GIOITINH else "Nu"

            # Vẽ nhãn
            text = f"{g_display}, {scope} (~{tuoi_hienthi})"
            ve_nhan_khung(frame, x, y, w, h, text)

            # Lưu cho popup
            bang_tom_tat.append((tid, g_display, scope, tuoi_hienthi, conf_nhom, float(prob)))

        # Hiển thị
        cv2.imshow("Du doan Tuoi + Gioi tinh (Realtime)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('a') and bang_tom_tat:
            # Popup tóm tắt
            s_lines = ["Kết quả dự đoán:"]
            for tid, g, scope, age, c_nhom, p_g in bang_tom_tat:
                s_lines.append(
                    f"ID {tid}: {g}, {scope}, ~{age} | Nhóm tuổi={c_nhom:.2f}, Giới tính={p_g:.2f}"
                )
            root = tk.Tk(); root.withdraw()
            messagebox.showinfo("Kết quả", "\n".join(s_lines))
            root.destroy()

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------
if __name__ == "__main__":
    main()
