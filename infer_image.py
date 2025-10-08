
import os, sys, cv2, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import img_to_array

# ===== CỜ HIỂN THỊ =====
SHOW_AGE        = True
SHOW_CONFIDENCE = False
BOX_THICKNESS   = 2

# ===== Cấu hình =====
MODEL_PATH   = "age_gender_model_v3.keras"
IMG_SIZE     = 224
MIN_FACE     = 40
AGE_GROUPS   = ['0-12','13-17','18-24','25-34','35-44','45-54','55-69','70+']

# ===== DNN face detector  =====
DNN_PROTO = "deploy.prototxt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
USE_DNN = os.path.exists(DNN_PROTO) and os.path.exists(DNN_MODEL)
if USE_DNN:
    net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
    print("[INFO] Dùng DNN phát hiện mặt.")
else:
    FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("[WARN] Thiếu DNN, dùng Haar fallback.")

def detect_faces(bgr):
    """Trả về list[(x,y,w,h), ...]"""
    H, W = bgr.shape[:2]
    faces = []
    if USE_DNN:
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr,(300,300)),1.0,(300,300),(104,177,123))
        net.setInput(blob)
        det = net.forward()
        for i in range(det.shape[2]):
            c = float(det[0,0,i,2])
            if c >= 0.5:
                x1,y1,x2,y2 = det[0,0,i,3:7]*[W,H,W,H]
                x1,y1,x2,y2 = map(int,[max(0,x1),max(0,y1),min(W-1,x2),min(H-1,y2)])
                w,h = x2-x1, y2-y1
                if w>=MIN_FACE and h>=MIN_FACE:
                    faces.append((x1,y1,w,h))
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        arr = FACE.detectMultiScale(gray,1.1,5,minSize=(MIN_FACE,MIN_FACE))
        if arr is not None and len(arr) > 0:
            faces = [tuple(map(int, box)) for box in arr]
    return faces  # luôn là list

def preprocess_face(bgr_face):
    rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb,(IMG_SIZE,IMG_SIZE))
    x = img_to_array(x)
    x = preprocess_input(x)
    return x

def draw_label(img, x, y, w, h, gender, agegrp=None, conf=None):
    color = (0,255,0) if gender=="Nam" else (255,0,255)
    cv2.rectangle(img,(x,y),(x+w,y+h),color,BOX_THICKNESS)

    line1 = gender
    line2 = None
    if SHOW_AGE and agegrp is not None:
        line2 = agegrp
    if SHOW_CONFIDENCE and conf is not None:
        line2 = (f"{line2} ({conf:.2f})" if line2 else f"{conf:.2f}")

    f1, f2 = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_SIMPLEX
    s1, s2  = 0.85, 0.65
    th1, th2 = 2, 2
    (tw1,th1_px), _ = cv2.getTextSize(line1, f1, s1, th1)
    tw2, th2_px = 0, 0
    if line2:
        (tw2,th2_px), _ = cv2.getTextSize(line2, f2, s2, th2)

    pad = 6
    box_w = max(tw1, tw2) + pad*2
    box_h = th1_px + (th2_px + 4 if line2 else 0) + pad*2

    bx, by = x, y - 8 - box_h
    if by < 0: by = y + h + 8
    bx = max(0, min(bx, img.shape[1]-box_w))

    overlay = img.copy()
    cv2.rectangle(overlay, (bx,by), (bx+box_w,by+box_h), (0,0,0), -1)
    img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    tx, ty = bx + pad, by + pad + th1_px
    cv2.putText(img, line1, (tx,ty), f1, s1, color, th1, cv2.LINE_AA)
    if line2:
        ty2 = ty + 4 + th2_px
        cv2.putText(img, line2, (tx,ty2), f2, s2, (255,255,255), th2, cv2.LINE_AA)

def main():
    model = load_model(MODEL_PATH, compile=False)
    print("[INFO] Đã load mô hình:", MODEL_PATH)

    path = sys.argv[1] if len(sys.argv) >= 2 else input("Nhập đường dẫn ảnh: ").strip()
    if not os.path.exists(path):
        print("❌ Không tìm thấy ảnh:", path); return

    img = cv2.imread(path)
    if img is None:
        print("❌ Không đọc được ảnh."); return

    boxes = detect_faces(img)
    # >>> FIX: kiểm tra rỗng bằng len()
    if boxes is None or len(boxes) == 0:
        print("⚠ Không thấy khuôn mặt nào.")
        cv2.imshow("Kết quả dự đoán qua ảnh", img)
        cv2.waitKey(0); cv2.destroyAllWindows()
        return

    patches, keep = [], []
    for (x,y,w,h) in boxes:
        roi = img[y:y+h, x:x+w]
        if roi.size == 0: continue
        patches.append(preprocess_face(roi))
        keep.append((x,y,w,h))

    X = np.stack(patches, axis=0)
    y_age, y_gender = model.predict(X, verbose=0)

    for i,(x,y,w,h) in enumerate(keep):
        age_idx = int(np.argmax(y_age[i])); agegrp = AGE_GROUPS[age_idx]
        p = float(y_gender[i][0])
        gender = "Nam" if p < 0.5 else "Nu"
        conf = (1-p) if gender=="Nam" else p
        draw_label(img, x,y,w,h, gender, agegrp, conf)

    cv2.imshow("Ket qua du doan tuoi & gioi tinh", img)
    cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
