import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
import pandas as pd

class VideoPlayer:
    def __init__(self, root, video_path, detections):
        self.root = root
        self.root.title("Video Player with Detections")

        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / self.fps)
        self.frame_idx = 0

        self.playing = False
        self.detections = detections.to_dict(orient="records")
        self.frame_label = ttk.Label(root)
        self.frame_label.pack()

        # ปุ่มควบคุม
        control_frame = ttk.Frame(root)
        control_frame.pack()

        self.play_button = ttk.Button(control_frame, text="▶ Play", command=self.play_video)
        self.play_button.pack(side="left", padx=5)

        self.pause_button = ttk.Button(control_frame, text="⏸ Pause", command=self.pause_video)
        self.pause_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(control_frame, text="⏹ Stop", command=self.stop_video)
        self.stop_button.pack(side="left", padx=5)

    def play_video(self):
        self.playing = True
        self.update_frame()

    def pause_video(self):
        self.playing = False

    def stop_video(self):
        self.playing = False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0

    def update_frame(self):
        if self.playing:
            ret, frame = self.cap.read()
            orig_w, orig_h = frame.shape[1], frame.shape[0]
            scale_x = 1280 / orig_w
            scale_y = 720 / orig_h
            frame = cv2.resize(frame, (1280, 720))
            if not ret:
                self.playing = False
                return

            self.frame_idx += 1
            timestamp = self.frame_idx / self.fps

            # --- วาด detection ที่ตรง timestamp ---
            for det in self.detections:
                if abs(det["timestamp"] - timestamp) < 0.09:  # เผื่อเวลา ±0.3s
                    # กรอบคน
                    x, y, w, h = det["x_person"], det["y_person"], det["w_person"], det["h_person"]
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                    # กรอบเสื้อผ้า
                    # x2 = int((det["x_clothing"] + det["x_person"]) * scale_x)
                    # y2 = int((det["y_clothing"] + det["y_person"]) * scale_y)
                    # w2 = int(det["w_clothing"] * scale_x)
                    # h2 = int(det["h_clothing"] * scale_y)
                    # cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (255,0,0), 2)

                    # Label
                    label = f"{det['track_id']} ({det['confidence']:.2f})"
                    cv2.putText(frame, label, (x, y-10+h),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # แปลง BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 360))

            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.frame_label.imgtk = img
            self.frame_label.config(image=img)

            self.root.after(self.delay, self.update_frame)

# ---------------- ใช้งาน ----------------
if __name__ == "__main__":
    # โหลด detections จาก JSON (แบบ list[dict])
    with open(r"E:\ALL_CODE\python\fashion-project\resources\result_prediction\clothing_detection\results_20250911_20250911_1.json", "r", encoding="utf-8") as f:
        detections = [json.loads(line) for line in f if line.strip()]  # NDJSON

    df = pd.DataFrame(detections)
    video_select = df[df['filename'] == "4p-c0-new.mp4"]
    print(len(video_select))
    root = tk.Tk()
    app = VideoPlayer(root, r"E:\ALL_CODE\python\fashion-project\resources\videos\4p-c0-new.mp4", video_select)
    root.mainloop()
