# ===============================
# HUẤN LUYỆN: TUỔI_NHÓM + TUỔI + GIỚI_TÍNH (đa nhiệm)
# ===============================
# - Đầu ra 1: tuoi_nhom (softmax 8 lớp) => dự đoán "scope" tuổi
# - Đầu ra 2: tuoi (sigmoid)           => hồi quy tuổi (0..1) -> *100
# - Đầu ra 3: gioitinh (sigmoid)       => 0=nam, 1=nữ
# ===============================

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# --------------------------
# Cấu hình
# --------------------------
KICH_THUOC_ANH = 64
BATCH_SIZE = 64
SO_EPOCH = 20
TUOI_TOI_DA = 100
DUONG_DAN_CSV = "data/labels.csv"
TEN_MO_HINH = "age_gender_model_1.keras"

# Định nghĩa các "scope" tuổi theo biên (tuổi thật nằm trong khoảng nào thì thuộc nhóm đó)
# biên = [0,12,17,24,34,44,54,64,100+]
NHOM_TUOI_BIEN = [(0,12),(13,17),(18,24),(25,34),(35,44),(45,54),(55,64),(65,150)]
SO_NHOM_TUOI = len(NHOM_TUOI_BIEN)

def xac_dinh_nhom_tuoi(tuoi: int) -> int:
    """Trả về chỉ số nhóm tuổi (0..7) dựa theo NHOM_TUOI_BIEN."""
    for i, (a,b) in enumerate(NHOM_TUOI_BIEN):
        if a <= tuoi <= b:
            return i
    return SO_NHOM_TUOI - 1  

# --------------------------
# Tạo dataset từ DataFrame 
# --------------------------
def tao_dataset_tu_df(df, kich_thuoc, batch_size, training=True):
    duongdan = df["filepath"].values
    tuoi_thuc = np.clip(df["age"].values.astype("float32"), 0, TUOI_TOI_DA)

    # chuẩn hoá tuổi 0..1 cho nhánh regression
    tuoi_scaled = (tuoi_thuc / TUOI_TOI_DA).astype("float32")
    gioitinh = df["gender"].values.astype("float32")

    # tạo nhãn "tuoi_nhom" từ tuổi thật
    tuoi_nhom_idx = np.array([xac_dinh_nhom_tuoi(int(t)) for t in tuoi_thuc], dtype="int32")
    tuoi_nhom_onehot = tf.one_hot(tuoi_nhom_idx, depth=SO_NHOM_TUOI)

    ds_duongdan = tf.data.Dataset.from_tensor_slices(duongdan)
    ds_tuoi_scaled = tf.data.Dataset.from_tensor_slices(tuoi_scaled)
    ds_tuoi_nhom = tf.data.Dataset.from_tensor_slices(tuoi_nhom_onehot)
    ds_gioitinh = tf.data.Dataset.from_tensor_slices(gioitinh)
    ds_nhan = tf.data.Dataset.zip((ds_tuoi_nhom, ds_tuoi_scaled, ds_gioitinh))

    def doc_anh(duong):
        anh = tf.io.read_file(duong)
        anh = tf.io.decode_image(anh, channels=3, expand_animations=False)
        anh = tf.image.convert_image_dtype(anh, tf.float32)
        anh = tf.image.resize(anh, (kich_thuoc, kich_thuoc), antialias=True)
        return anh

    ds = tf.data.Dataset.zip((ds_duongdan.map(doc_anh, num_parallel_calls=tf.data.AUTOTUNE), ds_nhan))

    def augment(anh, nhan):
        if training:
            anh = tf.image.random_flip_left_right(anh)
            anh = tf.image.random_brightness(anh, 0.1)
            anh = tf.image.random_contrast(anh, 0.8, 1.2)
        # đặt tên nhãn rõ ràng cho từng đầu ra
        return anh, {
            "tuoi_nhom": nhan[0],  # one-hot
            "tuoi": nhan[1],       # scalar 0..1
            "gioitinh": nhan[2]    # 0/1
        }

    if training:
        ds = ds.shuffle(4000, reshuffle_each_iteration=True)
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# --------------------------
# Kiến trúc mô hình 
# --------------------------
def tao_mo_hinh(kich_thuoc=64, so_nhom=8):
    dau_vao = layers.Input((kich_thuoc, kich_thuoc, 3))
    x = layers.SeparableConv2D(32, 3, padding="same", use_bias=False)(dau_vao); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.SeparableConv2D(64, 3, strides=2, padding="same", use_bias=False)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.SeparableConv2D(128, 3, strides=2, padding="same", use_bias=False)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # 1) Tuổi_nhóm (softmax)
    ra_tuoi_nhom = layers.Dense(so_nhom, activation="softmax", name="tuoi_nhom")(x)
    # 2) Tuổi (regression 0..1)
    ra_tuoi = layers.Dense(1, activation="sigmoid", name="tuoi")(x)
    # 3) Giới tính (binary)
    ra_gioitinh = layers.Dense(1, activation="sigmoid", name="gioitinh")(x)

    return keras.Model(dau_vao, [ra_tuoi_nhom, ra_tuoi, ra_gioitinh], name="age_group_age_gender")

# --------------------------
# Metric MAE tính theo NĂM cho nhánh tuổi
# --------------------------
class MAETuoiNam(keras.metrics.Metric):
    def __init__(self, max_age=100, name="mae_tuoi_nam", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_age = max_age
        self.mae = keras.metrics.MeanAbsoluteError()
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mae.update_state(y_true * self.max_age, y_pred * self.max_age, sample_weight)
    def result(self): return self.mae.result()
    def reset_states(self): self.mae.reset_states()

# --------------------------
# Huấn luyện
# --------------------------
if __name__ == "__main__":
    # Đọc CSV & chia train/val (giữ cân bằng giới tính)
    df = pd.read_csv(DUONG_DAN_CSV)
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["gender"], random_state=42)

    ds_train = tao_dataset_tu_df(train_df, KICH_THUOC_ANH, BATCH_SIZE, training=True)
    ds_val   = tao_dataset_tu_df(val_df,   KICH_THUOC_ANH, BATCH_SIZE, training=False)

    mo_hinh = tao_mo_hinh(KICH_THUOC_ANH, SO_NHOM_TUOI)
    mo_hinh.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "tuoi_nhom": "categorical_crossentropy",
            "tuoi": "mae",
            "gioitinh": "binary_crossentropy"
        },
        loss_weights={"tuoi_nhom": 1.0, "tuoi": 1.0, "gioitinh": 1.0},
        metrics={
            "tuoi_nhom": ["accuracy"],
            "tuoi": [MAETuoiNam(TUOI_TOI_DA)],
            "gioitinh": ["accuracy"]
        }
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_tuoi_nhom_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(TEN_MO_HINH, monitor="val_tuoi_nhom_accuracy", save_best_only=True)
    ]

    lich_su = mo_hinh.fit(ds_train, validation_data=ds_val, epochs=SO_EPOCH, callbacks=callbacks)
    mo_hinh.save(TEN_MO_HINH)
    print(" Đã lưu mô hình:", TEN_MO_HINH)
