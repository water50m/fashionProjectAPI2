import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# 1. โหลด YOLOv8 model (รุ่น nano สำหรับ realtime)
model = YOLO(r"E:\ALL_CODE\python\fashion-project\resources\models\yolov11n.pt") 

# 2. Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# 3. เปิดไฟล์วิดีโอ
video_path = r"E:\ALL_CODE\python\fashion-project\resources\videos\4p-c0-new.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. ทำ Object Detection
    results = model.predict(frame, verbose=False)[0]

    detections = []
    for box, score, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        class_id = int(cls)
        if class_id != 0:  # เฉพาะ person
            continue

        # แปลง tensor เป็น float / list ก่อนส่ง DeepSORT
        x1, y1, x2, y2 = box.tolist()
        conf = float(score)
        detections.append([x1, y1, x2, y2, conf])

    # 5. Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # 6. แสดงผล
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Person Tracking with DeepSORT", frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
