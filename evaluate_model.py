
# Đánh giá mô hình tuổi + nhóm tuổi + giới tính trên tập VAL
# ============================================

import os, re, math, sys, io
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error

# ---------------- CẤU HÌNH ----------------
DUONG_DAN_DATA = os.path.join("data", "utkface")
DUONG_DAN_MO_HINH = "age_gender_model_1.keras"

KICH_THUOC_ANH = 64
TUOI_TOI_DA = 100.0
BATCH_SIZE = 64
SEED = 42

# Nhóm tuổi (scope) – giống khi train/infer
NHOM_TUOI_BIEN = [(0,12),(13,17),(18,24),(25,34),(35,44),(45,54),(55,64),(65,150)]
NHOM_TUOI_TEN  = ["0-12","13-17","18-24","25-34","35-44","45-54","55-64","65+"]

# ---------------- HÀM TIỆN ÍCH ----------------
TEN_FILE_RE = re.compile(r"^(?P<age>\d+)_(?P<gender>[01])_")

def parse_label_from_filename(path):
    """
    Lấy (age, gender) từ tên file UTKFace: age_gender_...jpg
    Trả về (age:int, gender:int) hoặc None nếu không parse được.
    """
    name = os.path.basename(path)
    m = TEN_FILE_RE.match(name)
    if not m:
        return None
    try:
        age = int(m.group("age"))
        gender = int(m.group("gender"))
        if gender not in (0,1):  # 0=nam, 1=nữ
            return None
        # Áp trần tuổi giống lúc train
        age = max(0, min(int(age), int(TUOI_TOI_DA)))
        return age, gender
    except Exception:
        return None

def to_group_idx(age):
    """Chuyển tuổi thật -> chỉ số nhóm tuổi (0..7)."""
    for i, (lo, hi) in enumerate(NHOM_TUOI_BIEN):
        if lo <= age <= hi:
            return i
    return len(NHOM_TUOI_BIEN) - 1  # phòng hờ

def load_image_rgb(path, size=KICH_THUOC_ANH):
    """Đọc ảnh bằng OpenCV (BGR), resize, chuyển RGB, chuẩn hóa /255."""
    im = cv2.imread(path)
    if im is None:
        return None
    im = cv2.resize(im, (size, size))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype("float32") / 255.0
    return im

def batch_iter(paths, ages, genders, batch=BATCH_SIZE):
    """Sinh từng batch dữ liệu (X, y_age, y_gender, y_group)."""
    n = len(paths)
    for i in range(0, n, batch):
        xs, ya, yg, ygrou = [], [], [], []
        for p, a, g in zip(paths[i:i+batch], ages[i:i+batch], genders[i:i+batch]):
            im = load_image_rgb(p)
            if im is None:
                continue
            xs.append(im)
            ya.append(float(a) / TUOI_TOI_DA)  # nhãn tuổi chuẩn hóa 0..1
            yg.append(float(g))                # 0/1
            ygrou.append(to_group_idx(int(a))) # 0..7

        if not xs:
            continue
        X = np.stack(xs, axis=0)
        y_age = np.array(ya, dtype="float32")
        y_gender = np.array(yg, dtype="float32")
        y_group = np.array(ygrou, dtype="int64")
        yield X, y_age, y_gender, y_group

# ---------------- CHUẨN BỊ DỮ LIỆU ----------------
# Quét tất cả ảnh
all_paths = sorted(glob(os.path.join(DUONG_DAN_DATA, "*.jpg")))
records = []
for p in all_paths:
    lab = parse_label_from_filename(p)
    if lab is None:
        continue
    age, gender = lab
    records.append((p, age, gender))

df = pd.DataFrame(records, columns=["path", "age", "gender"])
if df.empty:
    print(" Không tìm thấy ảnh hợp lệ trong:", DUONG_DAN_DATA)
    sys.exit(1)

# Tách train/val (stratify theo giới tính để cân bằng)
train_df, val_df = train_test_split(
    df, test_size=0.15, stratify=df["gender"], random_state=SEED
)

print(f"Tổng ảnh: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")
print("Phân bố giới tính (val):\n", val_df["gender"].value_counts())

# ---------------- LOAD MODEL ----------------
try:
    model = load_model(DUONG_DAN_MO_HINH, compile=False)
except Exception as e:
    print(" Lỗi load mô hình:", e)
    sys.exit(1)

# Lưu summary ra file
os.makedirs("outputs", exist_ok=True)
summary_txt = os.path.join("outputs", "model_summary.txt")
with open(summary_txt, "w", encoding="utf-8") as f:
    s = io.StringIO()
    model.summary(print_fn=lambda x: s.write(x + "\n"))
    f.write(s.getvalue())
print(" Đã lưu kiến trúc mô hình vào:", summary_txt)

# ---------------- DỰ ĐOÁN TRÊN VAL ----------------
y_true_age, y_pred_age = [], []
y_true_gender, y_pred_gender = [], []
y_true_group,  y_pred_group  = [], []

for X, y_age, y_gender, y_group in batch_iter(
    val_df["path"].tolist(),
    val_df["age"].tolist(),
    val_df["gender"].tolist(),
    batch=BATCH_SIZE
):
    # model: [p_nhom (B,8), p_tuoi (B,1), p_gioi (B,1)]
    p_group, p_age, p_gender = model.predict(X, verbose=0)

    # Tuổi (0..1) -> năm
    pred_age_years = (p_age[:, 0] * TUOI_TOI_DA)
    true_age_years = (y_age * TUOI_TOI_DA)

    # Giới tính: <0.5 = Nam(0), >=0.5 = Nữ(1) (giống lúc infer)
    pred_gender_bin = (p_gender[:, 0] >= 0.5).astype(np.int64)

    # Nhóm tuổi: argmax softmax
    pred_group_cls = np.argmax(p_group, axis=1)

    # Gom lại
    y_true_age.extend(true_age_years.tolist())
    y_pred_age.extend(pred_age_years.tolist())

    y_true_gender.extend(y_gender.astype(np.int64).tolist())
    y_pred_gender.extend(pred_gender_bin.tolist())

    y_true_group.extend(y_group.tolist())
    y_pred_group.extend(pred_group_cls.tolist())

# ---------------- CHỈ SỐ ĐÁNH GIÁ ----------------
mae_age = mean_absolute_error(y_true_age, y_pred_age)
acc_gender = accuracy_score(y_true_gender, y_pred_gender)
acc_group  = accuracy_score(y_true_group,  y_pred_group)

print("\n==== KẾT QUẢ VAL ====")
print(f"MAE tuổi (năm): {mae_age:.3f}")
print(f"Accuracy giới tính: {acc_gender:.3f}")
print(f"Accuracy nhóm tuổi: {acc_group:.3f}")

# Confusion matrix
cm_gender = confusion_matrix(y_true_gender, y_pred_gender, labels=[0,1])
cm_group  = confusion_matrix(y_true_group,  y_pred_group,  labels=list(range(len(NHOM_TUOI_TEN))))

# Lưu CSV
pd.DataFrame(cm_gender, index=["Thật:Nam","Thật:Nữ"], columns=["Đoán:Nam","Đoán:Nữ"])\
  .to_csv(os.path.join("outputs","cm_gioitinh.csv"), encoding="utf-8-sig")

pd.DataFrame(cm_group, index=[f"Thật:{t}" for t in NHOM_TUOI_TEN],
             columns=[f"Đoán:{t}" for t in NHOM_TUOI_TEN])\
  .to_csv(os.path.join("outputs","cm_nhomtuoi.csv"), encoding="utf-8-sig")

print(" Đã lưu: outputs/cm_gioitinh.csv, outputs/cm_nhomtuoi.csv")

# (Tùy chọn) vẽ PNG nếu có matplotlib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns  # nếu không có cũng ổn; chỉ dùng nếu cài sẵn
    def plot_cm(cm, labels, title, out_path):
        plt.figure(figsize=(6,5))
        try:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        except Exception:
            plt.imshow(cm, cmap="Blues"); plt.colorbar()
            for (i,j),v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha='center', va='center')
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.yticks(range(len(labels)), labels)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    plot_cm(cm_gender, ["Nam","Nữ"], "Confusion Matrix – Giới tính", os.path.join("outputs","cm_gioitinh.png"))
    plot_cm(cm_group, NHOM_TUOI_TEN, "Confusion Matrix – Nhóm tuổi", os.path.join("outputs","cm_nhomtuoi.png"))
    print(" Đã lưu: outputs/cm_gioitinh.png, outputs/cm_nhomtuoi.png")
except Exception:
    print(" Không vẽ PNG (chưa cài matplotlib/seaborn). Bạn có thể: pip install matplotlib seaborn")

# Lưu classification_report cho nhóm tuổi & giới tính
rep_gender = classification_report(y_true_gender, y_pred_gender, target_names=["Nam","Nữ"])
rep_group  = classification_report(y_true_group,  y_pred_group,  target_names=NHOM_TUOI_TEN)

with open(os.path.join("outputs","report_gioitinh.txt"), "w", encoding="utf-8") as f:
    f.write(rep_gender)
with open(os.path.join("outputs","report_nhomtuoi.txt"), "w", encoding="utf-8") as f:
    f.write(rep_group)

print(" Đã lưu: outputs/report_gioitinh.txt, outputs/report_nhomtuoi.txt")
print("\n XONG. Xem thư mục outputs/ để lấy báo cáo.")
