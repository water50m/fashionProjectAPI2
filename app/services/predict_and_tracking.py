import cv2
from ultralytics import YOLO

import torch
from app.services.tracking_lib.use_func_track import usethis

# test tracking
def run_predict():
    # โหลดโมเดล
    model_custom = YOLO(r"E:\ALL_CODE\python\fashion-project\resources\models\yolo11m.pt").to("cuda")

    # เปิดวิดีโอ (0 = webcam)
    video_path = r"E:\ALL_CODE\nextjs\fashion_project_1\public\videos\uploads\4p-c0-new.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        # ขั้นแรก detect class [0] ด้วย pretrained model
        results = model_custom.predict(
            source=frame,
            classes=[0],     # เฉพาะ class 0 (เช่น person)
            conf=0.5,
            verbose=False
        )[0]

        tracks = usethis(results,frame,0)
        # วนลูปผลลัพธ์จาก pretrained
        for t in tracks:
  
            x1, y1, x2, y2 = [int(i) for i in t[:4]]  # bbox เป็น int
            track_id = int(t[4])                       # track_id
            conf = float(t[5])                         # confidence ยังเป็น float
            cls = int(t[6])
            frame_input_id = int(t[7])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0),2)
            cv2.putText(frame, f"ID {track_id} {conf}", (x1,y1+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)




        # แสดงวิดีโอ real-time
        cv2.imshow("tracking", frame)

        # กด q เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
