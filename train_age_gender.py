
import pandas as pd, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------- Cấu hình --------
KICH_THUOC_ANH = 224           
BATCH_SIZE = 64                
SO_EPOCH_WARMUP = 5            # Giai đoạn 1: freeze backbone -> train head
SO_EPOCH_FINETUNE = 25         # Giai đoạn 2: fine-tune một phần backbone
DUONG_DAN_CSV = "data/labels.csv"           # CSV: filepath, age, gender
TEN_MO_HINH = "age_gender_model_v3.keras"   # Tên file lưu mô hình

# Định nghĩa 8 nhóm tuổi (bao trùm toàn dải; giá trị age ngoài range sẽ bị clip về [0,100] ở dưới)
NHOM_TUOI_BIEN = [(0,12),(13,17),(18,24),(25,34),(35,44),(45,54),(55,64),(65,150)]
SO_NHOM_TUOI = len(NHOM_TUOI_BIEN)

def xac_dinh_nhom_tuoi(tuoi: int) -> int:
    """
    Map tuổi thật (int) -> chỉ số nhóm tuổi (0..SO_NHOM_TUOI-1).
    """
    for i,(a,b) in enumerate(NHOM_TUOI_BIEN):
        if a <= tuoi <= b: return i
    return SO_NHOM_TUOI - 1  # Phòng trường hợp rơi ngoài các khoảng (không xảy ra nếu đã clip)

# -------- Dataset --------
def tao_dataset_tu_df(df, kich_thuoc, batch_size, training=True):
    """
    Tạo tf.data.Dataset từ DataFrame:
      - Đọc ảnh từ đường dẫn, resize -> (kich_thuoc, kich_thuoc)
      - Chuẩn hoá theo EfficientNet (preprocess_input)
      - Tạo nhãn: (tuổi_nhom_onehot, giới_tính_scalar)
      - Thêm augment cơ bản khi training (flip, brightness/contrast/saturation nhẹ)
    """
    duongdan = df["filepath"].values
    # Bảo vệ dữ liệu đầu vào: tuổi chỉ lấy [0,100] để tránh nhãn lỗi
    tuoi_thuc = np.clip(df["age"].values.astype("int32"), 0, 100)
    gioitinh = df["gender"].values.astype("float32")  # kỳ vọng 0/1

    # One-hot nhóm tuổi
    tuoi_nhom_idx = np.array([xac_dinh_nhom_tuoi(int(t)) for t in tuoi_thuc], dtype="int32")
    tuoi_nhom_onehot = tf.one_hot(tuoi_nhom_idx, depth=SO_NHOM_TUOI)

    # Tách thành 3 dataset song song: đường dẫn, nhãn tuổi_nhóm, nhãn giới_tính
    ds_duongdan = tf.data.Dataset.from_tensor_slices(duongdan)
    ds_tuoi_nhom = tf.data.Dataset.from_tensor_slices(tuoi_nhom_onehot)
    ds_gioitinh = tf.data.Dataset.from_tensor_slices(gioitinh)
    ds_nhan = tf.data.Dataset.zip((ds_tuoi_nhom, ds_gioitinh))  # ((onehot_agegrp), gender)

    def doc_anh(p):
        """
        Đọc file ảnh -> decode -> resize -> float32 -> preprocess_input (chuẩn EfficientNet).
        expand_animations=False để tránh GIF nhiều frame.
        """
        x = tf.io.read_file(p)
        x = tf.io.decode_image(x, channels=3, expand_animations=False)
        x = tf.image.resize(x, (kich_thuoc, kich_thuoc), antialias=True)
        x = preprocess_input(tf.cast(x, tf.float32))  # scale/chuẩn hoá theo EfficientNet
        return x

    # Gộp (ảnh đã đọc/chuẩn hoá, nhãn)
    ds = tf.data.Dataset.zip((ds_duongdan.map(doc_anh, num_parallel_calls=tf.data.AUTOTUNE), ds_nhan))

    def augment(x, nhan):
        """
        Augment nhẹ khi training (giúp generalize):
          - lật ngang ngẫu nhiên
          - thay đổi saturation/contrast/brightness nhẹ
        """
        if training:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_saturation(x, 0.9, 1.1)
            x = tf.image.random_contrast(x, 0.9, 1.1)
            x = tf.image.random_brightness(x, 0.05)
        # Trả về dict khớp với tên các head của model
        return x, {"tuoi_nhom": nhan[0], "gioitinh": nhan[1]}

    # Shuffle lớn khi training; map->batch->prefetch để tối ưu I/O
    if training: ds = ds.shuffle(4000, reshuffle_each_iteration=True)
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)\
           .batch(batch_size)\
           .prefetch(tf.data.AUTOTUNE)
    return ds

# -------- Model --------
def tao_mo_hinh(kich_thuoc=224, so_nhom=8):
    """
    Xây dựng kiến trúc:
      - EfficientNetB0 (include_top=False, pretrained ImageNet)
      - GAP + Dropout(0.3)
      - 2 head:
          + 'tuoi_nhom' (Dense so_nhom, softmax)
          + 'gioitinh'  (Dense 1, sigmoid)
    """
    base = EfficientNetB0(include_top=False, weights="imagenet",
                          input_shape=(kich_thuoc, kich_thuoc, 3))
    base.trainable = False  # Warmup: đóng băng toàn bộ backbone

    inp = layers.Input((kich_thuoc, kich_thuoc, 3))
    x = base(inp, training=False)       # training=False để BatchNorm của backbone chạy ở inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    out_agegrp = layers.Dense(so_nhom, activation="softmax", name="tuoi_nhom")(x)
    out_gender = layers.Dense(1, activation="sigmoid", name="gioitinh")(x)
    return keras.Model(inp, [out_agegrp, out_gender], name="effb0_agegrp_gender")

if __name__ == "__main__":
    # Đọc CSV; đảm bảo có cột 'filepath','age','gender'
    df = pd.read_csv(DUONG_DAN_CSV)

    # Chia train/val (stratify theo 'gender' để cân bằng giới tính giữa 2 tập)
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["gender"], random_state=42)

    # Tạo tf.data.Dataset cho train/val
    ds_train = tao_dataset_tu_df(train_df, KICH_THUOC_ANH, BATCH_SIZE, training=True)
    ds_val   = tao_dataset_tu_df(val_df,   KICH_THUOC_ANH, BATCH_SIZE, training=False)

    # Khởi tạo model
    model = tao_mo_hinh(KICH_THUOC_ANH, SO_NHOM_TUOI)

    # ---- Warmup (freeze backbone) ----
    # Train phần head để ổn định các lớp trên trước khi fine-tune backbone
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss={"tuoi_nhom":"categorical_crossentropy","gioitinh":"binary_crossentropy"},
                  metrics={"tuoi_nhom":["accuracy"],"gioitinh":["accuracy"]})
    model.fit(ds_train, validation_data=ds_val, epochs=SO_EPOCH_WARMUP)

    # ---- Fine-tune (mở top 20 lớp cuối) ----
    # Lấy backbone từ model.layers (chú ý index: ở đây là EfficientNetB0)
    base = model.layers[1]  # EfficientNetB0 block
    base.trainable = True
    # Chỉ mở (trainable=True) ~20 lớp cuối để tránh overfit/đòi hỏi LR quá nhỏ
    for layer in base.layers[:-20]:
        layer.trainable = False

    # LR nhỏ hơn khi fine-tune để không phá hỏng trọng số pretrain
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss={"tuoi_nhom":"categorical_crossentropy","gioitinh":"binary_crossentropy"},
                  metrics={"tuoi_nhom":["accuracy"],"gioitinh":["accuracy"]})

    # Callback phổ biến:
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),  # dừng sớm
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),            # giảm LR khi không cải thiện
        keras.callbacks.ModelCheckpoint(TEN_MO_HINH, monitor="val_loss", save_best_only=True)      # lưu best theo val_loss
    ]

    # Huấn luyện giai đoạn 2
    model.fit(ds_train, validation_data=ds_val, epochs=SO_EPOCH_FINETUNE, callbacks=callbacks)

    # Lưu cuối cùng (dù ModelCheckpoint đã lưu best)
    model.save(TEN_MO_HINH)
    print(" Đã lưu mô hình:", TEN_MO_HINH)
