import cv2, os, numpy as np
from collections import OrderedDict
from tensorflow.keras.models import load_model

MODEL = "age_gender_model_v3.keras"
IMG_SIZE = 224
MIN_FACE = 80
THRESH_GENDER, MARGIN = 0.50, 0.10
TEXT_MALE, TEXT_FEMALE, TEXT_UNKNOWN = "Nam","Nu","Khong ro"
AGE_GROUPS = ["0-12","13-17","18-24","25-34","35-44","45-54","55-64","65+"]
ALPHA = 0.6

USE_DNN = os.path.exists("deploy.prototxt") and os.path.exists("res10_300x300_ssd_iter_140000.caffemodel")
if USE_DNN:
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt","res10_300x300_ssd_iter_140000.caffemodel")
    def detect_faces(bgr, conf=0.6):
        h,w=bgr.shape[:2]
        blob=cv2.dnn.blobFromImage(cv2.resize(bgr,(300,300)),1.0,(300,300),(104,177,123))
        net.setInput(blob); det=net.forward()
        boxes=[]
        for i in range(det.shape[2]):
            c=float(det[0,0,i,2])
            if c>=conf:
                x1,y1,x2,y2=det[0,0,i,3:7]*[w,h,w,h]
                x1,y1,x2,y2=map(int,[max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)])
                if x2-x1>=MIN_FACE and y2-y1>=MIN_FACE: boxes.append((x1,y1,x2-x1,y2-y1))
        return boxes
else:
    FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    def detect_faces(bgr, conf=0.6):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = FACE.detectMultiScale(gray,1.2,5,minSize=(MIN_FACE,MIN_FACE))
        return list(faces) if faces is not None else []

def prep(bgr):
    x=cv2.resize(bgr,(IMG_SIZE,IMG_SIZE))
    x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB).astype("float32")
    return np.expand_dims(x,0)

def ema(old,new,a=ALPHA): return new if old is None else old*(1-a)+new*a

def draw(frame,x,y,w,h,text):
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
    cv2.rectangle(frame,(x,y-8-th-6),(x+tw+6,y-4),(0,0,0),-1)
    cv2.putText(frame,text,(x+3,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

def main():
    model = load_model(MODEL, compile=False)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("❌ Không mở webcam"); return

    tracks = OrderedDict()

    while True:
        ok,frame = cap.read()
        if not ok: break
        faces = detect_faces(frame)

        new_tracks = OrderedDict()
        for i,(x,y,w,h) in enumerate(faces):
            roi = frame[y:y+h,x:x+w]
            xin = prep(roi)
            p_group,p_gender = model.predict(xin, verbose=0)

            # smoothing theo ID đơn giản (dựa index i)
            tid = i+1
            if tid in tracks:
                p_group[0] = ema(tracks[tid]["pg"], p_group[0])
                p_gender[0][0] = ema(tracks[tid]["gg"], float(p_gender[0][0]))

            idx = int(np.argmax(p_group[0])); scope = AGE_GROUPS[idx]
            prob = float(p_gender[0][0])
            gender = TEXT_UNKNOWN if abs(prob-THRESH_GENDER)<=MARGIN else (TEXT_MALE if prob<THRESH_GENDER else TEXT_FEMALE)

            new_tracks[tid] = {"pg": p_group[0], "gg": prob}
            draw(frame,x,y,w,h,f"{gender}, {scope}")

        tracks = new_tracks
        cv2.imshow("Realtime Age+Gender (v3)", frame)
        if (cv2.waitKey(1)&0xFF)==ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
