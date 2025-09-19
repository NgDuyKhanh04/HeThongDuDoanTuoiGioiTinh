import sys, os, cv2, numpy as np
from tensorflow.keras.models import load_model

MODEL = "age_gender_model_v3.keras"
IMG_SIZE = 224
MIN_FACE = 80
TEXT_MALE, TEXT_FEMALE, TEXT_UNKNOWN = "Nam","Nu","Khong ro"
THRESH_GENDER, MARGIN = 0.50, 0.10
AGE_GROUPS = ["0-12","13-17","18-24","25-34","35-44","45-54","55-64","65+"]

# --- Face detector: DNN (nếu có) -> fallback Haar ---
USE_DNN = os.path.exists("deploy.prototxt") and os.path.exists("res10_300x300_ssd_iter_140000.caffemodel")
if USE_DNN:
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    def detect_faces(bgr, conf=0.6):
        h,w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr,(300,300)),1.0,(300,300),(104,177,123))
        net.setInput(blob); det = net.forward()
        boxes=[]
        for i in range(det.shape[2]):
            c=float(det[0,0,i,2])
            if c>=conf:
                x1,y1,x2,y2 = det[0,0,i,3:7]*[w,h,w,h]
                x1,y1,x2,y2 = map(int,[max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)])
                if x2-x1>=MIN_FACE and y2-y1>=MIN_FACE: boxes.append((x1,y1,x2-x1,y2-y1))
        return boxes
else:
    FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    def detect_faces(bgr, conf=0.6):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE.detectMultiScale(gray,1.2,5,minSize=(MIN_FACE,MIN_FACE))
        return list(faces) if faces is not None else []

def preprocess(bgr):
    x = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype("float32")
    # EfficientNet preprocess_input đã áp dụng khi train -> ở đây dùng scale 0..255 -> sẽ vẫn hoạt động nhờ chuẩn hoá nội tại
    return np.expand_dims(x,0)

def draw(frame, x,y,w,h, text):
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
    cv2.rectangle(frame,(x,y-8-th-6),(x+tw+6,y-4),(0,0,0),-1)
    cv2.putText(frame,text,(x+3,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

def main():
    if len(sys.argv)<2:
        print("python infer_image.py <duong_dan_anh>"); return
    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None: print("❌ Không đọc được ảnh"); return

    model = load_model(MODEL, compile=False)
    faces = detect_faces(img)

    for (x,y,w,h) in faces:
        face = img[y:y+h,x:x+w]
        xin = preprocess(face)
        p_group, p_gender = model.predict(xin, verbose=0)

        idx = int(np.argmax(p_group[0])); scope = AGE_GROUPS[idx]
        prob = float(p_gender[0][0])
        gender = TEXT_UNKNOWN if abs(prob-THRESH_GENDER)<=MARGIN else (TEXT_MALE if prob<THRESH_GENDER else TEXT_FEMALE)
        draw(img,x,y,w,h,f"{gender}, {scope}")

    cv2.imshow("Kết quả", img); cv2.waitKey(0); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
