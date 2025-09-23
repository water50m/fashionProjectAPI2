import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
import os


    # ถ้า __file__ ไม่มีค่า ให้ใช้ cwd แทน
ROOT = Path.cwd().resolve()

print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from app.services.config_service import load_config

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


            # --- วาด detection ที่ตรง timestamp ---
            for det in self.detections:
                if det["frame"] == self.frame_idx:
                    # กรอบคน
                    x, y, w, h = det["x_person"], det["y_person"], det["w_person"], det["h_person"]
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    # Label
                    label = f"{det['track_id']} ({det['confidence']})"
                    cv2.putText(frame, label, (x, y+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # กรอบเสื้อผ้า
                    x2 = int((det["x_clothing"] + det["x_person"]) * scale_x)
                    y2 = int((det["y_clothing"] + det["y_person"]) * scale_y)
                    w2 = int(det["w_clothing"] * scale_x)
                    h2 = int(det["h_clothing"] * scale_y)
                    cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (255,0,0), 2)

                    # Label
                    label = f"({det['class_name']})"
                    cv2.putText(frame, label, (x2, y2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # แปลง BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 360))

            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.frame_label.imgtk = img
            self.frame_label.config(image=img)

            self.root.after(self.delay, self.update_frame)

config = load_config()
class GraphPlayer:
    def __init__(self, detections, track_id=None, show_clothing=False):
        self.show_cloth = show_clothing
        self.track = track_id
        self.result = detections
        # self.graph = self.plot()
        self.playing = False

        self.input = 'start'#input('plat? \n')
        

        self.video_path =  config.get("VIDEO_PATH")
        self.stop = False

        if self.input:
            self.input = None
            self.play()

    def play(self):
        cv2.destroyAllWindows()  
        self.stop = True
        plt.close('all')
        print('random new')

        self.playing = True
        self.selet_data()
        self.input = input('...:')
    
    def select_best(self,weight,class_name):
        # argsort จะ sort index ตามค่า
        top_idx = weight.argsort()[::-1][:2]  # เอา index 2 ตัวที่มากที่สุด

        class_top1, class_top2 = class_name[top_idx[0]], class_name[top_idx[1]]
        return class_top1,class_top2

    def calculate_weight(self, list_data, weight):
        result = 0
        for i,item in enumerate(list_data):
            list_data = np.array(item)
            normalized1 = ((list_data - list_data.min()) / (list_data.max() - list_data.min())) * weight[i]
            result+= normalized1
           
        return result
    
    def selet_data(self):
        group_track = self.result.groupby('track_id')
        group_names = list(group_track.groups.keys())  # ['A','B','C']
        random_group_name = np.random.choice(group_names)
        if self.track is not None:
            random_group_name = self.track
        data_random_group = group_track.get_group(random_group_name)
        group_class = data_random_group.groupby('class_name')


        y = []
        x = []
        sum_confidence = []
        mean_confidence = []
        for group_name, group_df in group_class:
            size = group_df.size
            sum_conf = group_df['confidence'].sum()/10
            mean_conf = (float(sum_conf)/float(size/100))
            sum_confidence.append(sum_conf)
            mean_confidence.append(mean_conf)
            y.append(size)
            x.append(group_name)

        

        weights_cal = self.calculate_weight([sum_confidence,y,mean_confidence],[0.7,0.3,1.3])
        top2_weight = sorted(weights_cal, reverse=True)[:2]
        self.plot(weights_cal,top2_weight,sum_confidence,mean_confidence,random_group_name,x,y)
        self.video_player(data_random_group)  
        

    def plot(self,weights_cal,top2_weight,sum_confidence,mean_confidence,random_group_name,x,y):


        plt.plot(x, y, marker="o", linestyle="--", color="red")
        # แสดงค่า y บนแต่ละจุด
        offset = max(y)/20 

        for i, value in enumerate(y):
            color_= 'red' if weights_cal[i] in top2_weight else 'green'
            plt.text(x[i], value + 1, str(value)+'N', ha='center')  
            plt.text(x[i], value + offset, str(round(sum_confidence[i],2))+'s', ha='center')  
            plt.text(x[i], value + offset + offset, str(round(mean_confidence[i],2))+'m', ha='center')
            plt.text(x[i], value + offset + offset + offset, str(round(weights_cal[i],2))+'w', ha='center', color=color_)
            

            # ha='center' = จัดให้อยู่ตรงกลางแกน x
            # value + 0.2 = ยกตัวเลขให้สูงกว่าจุดเล็กน้อย
        plt.xlabel("founded class")
        plt.ylabel("count item in class")
        plt.title("ID :"+str(random_group_name))
        plt.grid(True)
        # หมุน label ของ x-axis
        plt.xticks(rotation=60)   # 45 องศา เฉียง
        # ถ้าต้องการแนวตั้งจริงๆ ใช้ rotation=90

        plt.tight_layout()  # ปรับ layout ให้อ่าน label ได้ไม่โดนตัด
        plt.show(block=False) 
        
    
    def video_player(self,data_random_group):
        print('precessing data')

        group_frame = data_random_group.groupby('frame')

        filename = data_random_group['filename'].tolist()
        frames = data_random_group["frame"].tolist()
        crops = data_random_group[["x_person", "y_person", "w_person", "h_person"]].values.tolist()
        class_name = data_random_group['class_name'].tolist()
        Ids = data_random_group['track_id'].tolist()
        print("play video")


        self.stop = False 
        video = self.video_path + filename[0]
        cap = cv2.VideoCapture(video)
        

        while True:  # วนซ้ำไม่สิ้นสุด
            for frame_idx, detail in group_frame:
                # เลื่อนไปยัง frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"ไม่สามารถอ่าน frame {frame_idx} ได้")
                    continue

                # loop ผ่าน row ของ group
                for idx, item in detail.iterrows():
                    x, y, w, h = item['x_person'], item['y_person'], item['w_person'], item['h_person']
                    class_name = item['class_name']

                    # วาด rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # ใส่ text
                    # cv2.putText(frame, f"ID:{item['track_id']}", (x, y+20),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if self.show_cloth:
                        xc, yc, wc, hc = item['x_clothing'], item['y_clothing'], item['w_clothing'], item['h_clothing']
                        # วาด rectangle
                        cv2.rectangle(frame, (x+xc, y+yc), (x+wc+xc, y+hc+yc), (0, 0, 255), 2)
                        cv2.putText(frame, f"{class_name}", (x+xc, y+yc+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

                # แสดง frame
                cv2.imshow("Video", frame)

                # รอ 30ms ระหว่าง frame, กด 'q' เพื่อออก
                if cv2.waitKey(400) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()  # ออกจากโปรแกรม

            # เมื่อ loop group_frame จบ → จะเริ่มใหม่อัตโนมัติ



class SummaryData:
    def __init__(self,root, detections):
        self.root = root
        self.data = detections
        self.summary()

    def run(self):
        directive = input("start ")
        directive = 'random'
        if directive:
             self.summary()
             
    def summary(self):
        group_track = self.data.groupby('track_id')
        group_names = list(group_track.groups.keys())  # ['A','B','C']
        random_group_name = np.random.choice(group_names)
        data_random_group = group_track.get_group(random_group_name)
        group_class = data_random_group.groupby('class_name')
        summary_data = pd.DataFrame(columns=['class_name','total','0.3','0.4','0.5','0.6','0.7'])
        for name,data in group_class:
            new_row = {
                'class_name': name,
                'total': data.shape[0],
                '0.3': (data['confidence'] > 0.3).sum(),
                '0.4': (data['confidence'] > 0.4).sum(),
                '0.5': (data['confidence'] > 0.5).sum(),
                '0.6': (data['confidence'] > 0.6).sum(),
                '0.7': (data['confidence'] > 0.7).sum()
            }

            summary_data = pd.concat([summary_data, pd.DataFrame([new_row])], ignore_index=True)
        print("ID :",random_group_name)
        print(summary_data)
        self.run()



             


# ---------------- ใช้งาน ----------------
if __name__ == "__main__":
    # โหลด detections จาก JSON (แบบ list[dict])
    with open(r"E:\ALL_CODE\python\fashion-project\resources\result_prediction\clothing_detection\results_clothing_detection_20250917_2.json", "r", encoding="utf-8") as f:
        detections = json.load(f) 
    df = pd.DataFrame(detections)
    video_select = df[df['filename'] == "4p-c0-new.mp4"]

    print(len(video_select))
    app = SummaryData('root',video_select)
    # app = GraphPlayer(video_select,59,True)
    # root = tk.Tk()
    
    
    # app = VideoPlayer(root, r"E:\ALL_CODE\python\fashion-project\resources\videos\4p-c0-new.mp4", video_select)
    # root.mainloop()
    # [67,42,40,69]
    # 67 frame ที่เห็นเต็มตัวมีน้อยกว่า
    # 42 มีส่วนที่โดนบังและ มีหลาย frame ที่เหลือครึ่งตัว
    # 40 มุมด้านข้าง + ระยะไกล + ภาพไม่ชัด  แขนยกขึ้น
    # 17,69 เห็นแต่หฟัว
    # 32 มือบังส่วนที่เชื่อมต่อระหว่างเสื้อแลักางเกงทำให้มองเห็นเป็น dress
    # 27,46 เอามือขวั้ยหลังแล้วเมื่อมองด้านหน้าจะเห็นมือแค่ครึ่งแขน
    # 24 จังหวะคนซ้อนกันพอดีทำให้ เห็นเป็นอย่างอื่น(dress ใน track นี้)
    # 59 long sleave top และ long sleave outwear 
