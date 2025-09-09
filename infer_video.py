
# Dự đoán TUỔI + GIỚI TÍNH trên VIDEO (không dùng webcam/URL)
# ==============================

import os, sys, cv2, numpy as np
from collections import OrderedDict
from tensorflow.keras.models import load_model

# ---- Cấu hình ----
DUONG_DAN_MO_HINH = "age_gender_model_1.keras"
KICH_THUOC_ANH = 64
TUOI_TOI_DA = 100.0
MIN_FACE_SIZE = 80  # px

# Nhãn hiển thị
TEXT_MALE = "Nam"
TEXT_FEMALE = "Nu"
TEXT_UNKNOWN = "Khong rõ"

# Ngưỡng quyết định
NGUONG_GIOITINH = 0.50
BIEN_DO_GIOITINH = 0.10
TIN_CAY_NHOM = 0.40

NHOM_TUOI_TEN  = ["0-12","13-17","18-24","25-34","35-44","45-54","55-64","65+"]

# Làm mượt EMA theo ID khuôn mặt
ALPHA_TUOI = 0.6
ALPHA_GIOITINH = 0.6
ALPHA_NHOM = 0.6

# Định dạng video hợp lệ
EXTS_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ---------------- utils ----------------
def _clean(s: str) -> str:
    if s is None: return ""
    return s.strip().strip('"').strip("'").strip()

def _resolve_video_path(user_in: str) -> str:

    p = _clean(user_in)
    if not p:
        return p

    # Nếu người dùng lỡ dán cả "python infer_video.py xxx", lấy token cuối
    parts = p.split()
    cand = _clean(parts[-1])

    # Nếu đã tồn tại -> dùng luôn
    if os.path.exists(cand):
        return cand

    # Nếu chỉ là tên file -> ghép với thư mục chương trình (Age_Gender)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.dirname(cand) == "":
        trial = os.path.join(script_dir, cand)
        if os.path.exists(trial):
            return trial

    # Không tìm thấy -> trả về nguyên bản (để báo lỗi)
    return cand

def _is_video_ext(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in EXTS_VIDEO

def _preprocess(face_bgr):
    im = cv2.resize(face_bgr, (KICH_THUOC_ANH, KICH_THUOC_ANH))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype("float32")/255.0
    return np.expand_dims(im, 0)

def _annotate(frame, x,y,w,h, text):
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y-8-th-6), (x+tw+6, y-4), (0,0,0), -1)
    cv2.putText(frame, text, (x+3, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def _ema(old, new, a):
    if old is None: return new
    return old*(1-a) + new*a

def _assign_ids(faces, tracks, nguong=110):
    centers = [(x+w//2, y+h//2) for (x,y,w,h) in faces]
    used = set()
    new_tracks = OrderedDict()
    for tid, st in tracks.items():
        if not centers: continue
        d = [abs(cx-st['cx']) + abs(cy-st['cy']) for (cx,cy) in centers]
        j = int(np.argmin(d))
        if j in used or d[j] > nguong:
            continue
        cx, cy = centers[j]
        new_tracks[tid] = {'cx':cx,'cy':cy,'bbox':faces[j],
                           'p_nhom':st.get('p_nhom'),
                           'tuoi':st.get('tuoi'),
                           'gioi':st.get('gioi')}
        used.add(j)
    next_id = (max(tracks.keys())+1) if len(tracks) else 1
    for k,(cx,cy) in enumerate(centers):
        if k in used: continue
        new_tracks[next_id] = {'cx':cx,'cy':cy,'bbox':faces[k],
                               'p_nhom':None,'tuoi':None,'gioi':None}
        next_id += 1
    return new_tracks

# ---------------- main ----------------
def main():
    # 1) Lấy đường dẫn video
    if len(sys.argv) >= 2:
        raw = " ".join(sys.argv[1:])
    else:
        print(" python infer_video.py <ten_file_or_duong_dan_video>")
        try:
            raw = input("Nhập TÊN FILE  hoặc đường dẫn video: ")
        except EOFError:
            raw = ""

    video_path = _resolve_video_path(raw)

    if not os.path.exists(video_path):
        print(" Không thấy file:", video_path); return
    if not _is_video_ext(video_path):
        print(" File không phải video. Dùng infer_image.py cho ảnh."); return

    # 2) Load model
    try:
        model = load_model(DUONG_DAN_MO_HINH, compile=False)
    except Exception as e:
        print(" Lỗi load mô hình:", e); return

    # 3) Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Không mở được video:", video_path); return

    # 4) VideoWriter (ghi cùng thư mục với nguồn, thêm _out.mp4)
    base, _ = os.path.splitext(video_path)
    out_path = base + "_out.mp4"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1: fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    print(f"💾 Ghi ra: {out_path} | {w}x{h} @ {fps:.1f}fps")
    print("Nhấn 'q' để dừng sớm.")

    # 5) Dò mặt + dự đoán
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    tracks = OrderedDict()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_np = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        faces = list(faces_np) if faces_np is not None else []
        tracks = _assign_ids(faces, tracks, nguong=110)

        for tid, st in tracks.items():
            x,y,wf,hf = st['bbox']
            if wf < MIN_FACE_SIZE or hf < MIN_FACE_SIZE: 
                continue
            roi = frame[y:y+hf, x:x+wf]
            if roi.size == 0: 
                continue

            xin = _preprocess(roi)
            try:
                ypred = model.predict(xin, verbose=0)
            except Exception as e:
                print(" Lỗi dự đoán:", e); 
                continue

            if not isinstance(ypred, (list,tuple)) or len(ypred)!=3:
                continue

            p_nhom, p_tuoi, p_gioi = ypred

            # Nhóm tuổi
            vec = p_nhom[0]
            st['p_nhom'] = vec if st['p_nhom'] is None else _ema(st['p_nhom'], vec, ALPHA_NHOM)
            idx = int(np.argmax(st['p_nhom']))
            conf = float(st['p_nhom'][idx])
            scope = NHOM_TUOI_TEN[idx] if conf >= TIN_CAY_NHOM else f"{NHOM_TUOI_TEN[idx]}?"

            # Tuổi
            age = float(p_tuoi[0][0]) * TUOI_TOI_DA
            st['tuoi'] = age if st['tuoi'] is None else _ema(st['tuoi'], age, ALPHA_TUOI)
            age_show = int(round(st['tuoi']))

            # Giới tính
            prob = float(p_gioi[0][0])
            st['gioi'] = prob if st['gioi'] is None else _ema(st['gioi'], prob, ALPHA_GIOITINH)
            p = st['gioi']
            if abs(p - NGUONG_GIOITINH) <= BIEN_DO_GIOITINH:
                g = TEXT_UNKNOWN
            else:
                g = TEXT_MALE if p < NGUONG_GIOITINH else TEXT_FEMALE

            _annotate(frame, x,y,wf,hf, f"{g}, {scope} (~{age_show})")

        out.write(frame)
        cv2.imshow("Age & Gender - VIDEO", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release(); out.release()
    cv2.destroyAllWindows()
    print(" Hoàn tất. File đã ghi:", out_path)

if __name__ == "__main__":
    main()
