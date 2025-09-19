# ===============================
# TRAIN v3: NHÓM TUỔI + GIỚI TÍNH (EfficientNetB0)
# ===============================
import pandas as pd, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------- Cấu hình --------
KICH_THUOC_ANH = 224
BATCH_SIZE = 64
SO_EPOCH_WARMUP = 5        # giai đoạn 1 (freeze backbone)
SO_EPOCH_FINETUNE = 25     # giai đoạn 2 (fine-tune top layers)
DUONG_DAN_CSV = "data/labels.csv"
TEN_MO_HINH = "age_gender_model_v3.keras"

NHOM_TUOI_BIEN = [(0,12),(13,17),(18,24),(25,34),(35,44),(45,54),(55,64),(65,150)]
SO_NHOM_TUOI = len(NHOM_TUOI_BIEN)

def xac_dinh_nhom_tuoi(tuoi: int) -> int:
    for i,(a,b) in enumerate(NHOM_TUOI_BIEN):
        if a <= tuoi <= b: return i
    return SO_NHOM_TUOI - 1

# -------- Dataset --------
def tao_dataset_tu_df(df, kich_thuoc, batch_size, training=True):
    duongdan = df["filepath"].values
    tuoi_thuc = np.clip(df["age"].values.astype("int32"), 0, 100)
    gioitinh = df["gender"].values.astype("float32")

    tuoi_nhom_idx = np.array([xac_dinh_nhom_tuoi(int(t)) for t in tuoi_thuc], dtype="int32")
    tuoi_nhom_onehot = tf.one_hot(tuoi_nhom_idx, depth=SO_NHOM_TUOI)

    ds_duongdan = tf.data.Dataset.from_tensor_slices(duongdan)
    ds_tuoi_nhom = tf.data.Dataset.from_tensor_slices(tuoi_nhom_onehot)
    ds_gioitinh = tf.data.Dataset.from_tensor_slices(gioitinh)
    ds_nhan = tf.data.Dataset.zip((ds_tuoi_nhom, ds_gioitinh))

    def doc_anh(p):
        x = tf.io.read_file(p)
        x = tf.io.decode_image(x, channels=3, expand_animations=False)
        x = tf.image.resize(x, (kich_thuoc, kich_thuoc), antialias=True)
        x = preprocess_input(tf.cast(x, tf.float32))  # chuẩn hoá theo EfficientNet
        return x

    ds = tf.data.Dataset.zip((ds_duongdan.map(doc_anh, num_parallel_calls=tf.data.AUTOTUNE), ds_nhan))

    def augment(x, nhan):
        if training:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_saturation(x, 0.9, 1.1)
            x = tf.image.random_contrast(x, 0.9, 1.1)
            x = tf.image.random_brightness(x, 0.05)
        return x, {"tuoi_nhom": nhan[0], "gioitinh": nhan[1]}

    if training: ds = ds.shuffle(4000, reshuffle_each_iteration=True)
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -------- Model --------
def tao_mo_hinh(kich_thuoc=224, so_nhom=8):
    base = EfficientNetB0(include_top=False, weights="imagenet",
                          input_shape=(kich_thuoc, kich_thuoc, 3))
    base.trainable = False  # warmup

    inp = layers.Input((kich_thuoc, kich_thuoc, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    out_agegrp = layers.Dense(so_nhom, activation="softmax", name="tuoi_nhom")(x)
    out_gender = layers.Dense(1, activation="sigmoid", name="gioitinh")(x)
    return keras.Model(inp, [out_agegrp, out_gender], name="effb0_agegrp_gender")

if __name__ == "__main__":
    df = pd.read_csv(DUONG_DAN_CSV)
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["gender"], random_state=42)

    ds_train = tao_dataset_tu_df(train_df, KICH_THUOC_ANH, BATCH_SIZE, training=True)
    ds_val   = tao_dataset_tu_df(val_df,   KICH_THUOC_ANH, BATCH_SIZE, training=False)

    model = tao_mo_hinh(KICH_THUOC_ANH, SO_NHOM_TUOI)

    # ---- Warmup (freeze backbone) ----
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss={"tuoi_nhom":"categorical_crossentropy","gioitinh":"binary_crossentropy"},
                  metrics={"tuoi_nhom":["accuracy"],"gioitinh":["accuracy"]})
    model.fit(ds_train, validation_data=ds_val, epochs=SO_EPOCH_WARMUP)

    # ---- Fine-tune (mở top 20 lớp cuối) ----
    base = model.layers[1]  # EfficientNetB0 block
    base.trainable = True
    for layer in base.layers[:-20]:  # khoá bớt, chỉ fine-tune phần trên
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss={"tuoi_nhom":"categorical_crossentropy","gioitinh":"binary_crossentropy"},
                  metrics={"tuoi_nhom":["accuracy"],"gioitinh":["accuracy"]})

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint(TEN_MO_HINH, monitor="val_loss", save_best_only=True)
    ]
    model.fit(ds_train, validation_data=ds_val, epochs=SO_EPOCH_FINETUNE, callbacks=callbacks)
    model.save(TEN_MO_HINH)
    print("✅ Đã lưu mô hình:", TEN_MO_HINH)
