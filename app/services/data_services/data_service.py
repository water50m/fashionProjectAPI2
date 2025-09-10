import json
import os
import csv
from app.services.config_service import load_config
from app.services.color_service import ColorCheck
import cv2
import pandas as pd
import datetime
import uuid

CONFIG = load_config()

class DataManager:
    def __init__(self, config=None):
        self.config = CONFIG
        self.color_manage = ColorCheck()
    # ===== Get all video names =====
    def get_all_video_name(self, path=None):
        try:
            videos = []
            video_path = path or self.config.get("VIDEO_PATH", "")
            if os.path.exists(video_path):
                for file in os.listdir(video_path):
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        videos.append({
                            "file_name": file,
                            "file_path": os.path.join(video_path, file),
                            "predictionStrategyClothing": self.config.get("JSON_RESULT_PREDICT_CLOTHING", ""),
                            "predictionStrategyPerson": self.config.get("JSON_RESULT_PREDICT_PERSON", "")
                        })
            print(f"Found videos: {videos}")
            return sorted(videos, key=lambda x: x['file_name'])
        except Exception as e:
            print(f"[get_all_video_name] error: {e}")
            return []
    
    def get_dot_pt_file(self, path=None):
        try:
            fileList = []
  
            file_path = "../"+self.config.get("AI_MODEL_PATH"," ") 
   
            if os.path.exists(file_path):

                for file in os.listdir(file_path):
                    if file.lower().endswith(('.pt')):
                        fileList.append(file)
      
                return fileList
        except Exception as e:
            print(f"[get_dot_pt_file] error: {e}")
            return []

    # ===== manage  JSON =====
    def load_json(self, file_path):
        try:
            if not os.path.exists(file_path):
                return []
            with open(file_path, 'r', encoding='utf-8') as f:
                data =  json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data
            else:
                return []
        except Exception as e:
            print(f"[load_json] error: {e}")
            return []
        
    def load_last_json(self, file_path):
        try:
            if not os.path.exists(file_path):
                return []
            with open(file_path, 'r', encoding='utf-8') as f:
                data =  json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data[-1]
            else:
                return []
        except Exception as e:
            print(f"[load_json] error: {e}")
            return []


    def save_json(self, file_path, data):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"[save_json] Saved JSON to {file_path}")
        except Exception as e:
            print(f"[save_json] error: {e}")


    def update_json(self, file_path, new_data):
        try:
            data = self.load_json(file_path)
            if not isinstance(data, dict):
                data = {}
            data.update(new_data)
            self.save_json(file_path, data)
            print(f"[update_json] Updated JSON at {file_path}")
        except Exception as e:
            print(f"[update_json] error: {e}")

    def update_result_to_json(self, filepath, new_data: list[dict]):
            # โหลดข้อมูลเดิม
        with open(filepath, "a", encoding="utf-8") as f:
            lines = [json.dumps(item, ensure_ascii=False) for item in new_data]
            f.write("\n".join(lines) + "\n")


    # ===== Save CSV =====
    def save_csv(self, file_path, data):
        try:
            with open(file_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(data)
            print(f"[save_csv] Saved CSV to {os.path.abspath(file_path)}")
        except Exception as e:
            print(f"[save_csv] error: {e}")

    # ===== Load CSV =====
    def load_csv(self, file_path):
        try:
            loaded_data = []
            with open(file_path, mode="r", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    loaded_data.append(row)
            return loaded_data
        except Exception as e:
            print(f"[load_csv] error: {e}")
            return []
    
    def get_result_detail(self, data):
        try:
            print("data.process_id",data.process_id)
            print("data.filename",data.filename)
            if data.process_id and data.process_id != "undefined" and data.filename:
                print("data.process_id",True)
                df= self.load_csv(self.config.get("PROCESSED_LOG", ""))
                df = df[df['process_id'] == data.process_id]
    
                videoDetail={
                    "file_name":data.filename,
                    "file_path":self.config.get("UPLOAD_VIDEO_PATH", "")+data.filename,
                    "predictionStrategyClothing":df.clothing_detection_result_path.values[0],
                    "predictionStrategyPerson":df.person_detection_result_path.values[0]
                }
                return videoDetail

            elif data.filename:
                print("data.filename",True)
                videoDetail={
                    "file_name":data.filename,
                    "file_path":self.config.get("VIDEO_PATH", "")+data.filename,
                    "predictionStrategyClothing":self.config.get("JSON_RESULT_PREDICT_CLOTHING", ""),
                    "predictionStrategyPerson":self.config.get("JSON_RESULT_PREDICT_PERSON", "")
                }
                return videoDetail
            
        except Exception as e:
            print(f"[get_video_detail] error: {e}")

   
    def analyze_result_predict_data(self,data):
        data = self.load_json(data.predictionStrategyClothing)
        try:
            cap = cv2.VideoCapture(data.file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            data['video_duration'] = duration
            cap.release()
        except:
            pass

        return data
    
    def get_log_selcet_video_detection(self):
        try:
            path = self.config.get("PROCESSED_LOG", "")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df = df[df["process_type"] == "video_select_detection"]
                return df.to_dict(orient="records")
            else:
                return []
        except Exception as e:
            print(f"get_log_selcet_video_detection is error : {e}")

    def write_log(self, filename, cfg,predict_id,process_type,save_name):
        try:
            os.makedirs(os.path.dirname(cfg["PROCESSED_LOG"]), exist_ok=True) if os.path.dirname(cfg["PROCESSED_LOG"]) else None

            # header ที่เราต้องการ
            headers = ["predict_id","process_id","filename", "datetime", "person_detection_result_path", "clothing_detection_result_path", "save_result_name","process_type"]
            
            # ถ้าไฟล์ยังไม่เคยมี -> สร้างใหม่พร้อม header
            if not os.path.exists(cfg["PROCESSED_LOG"]):
                with open(cfg["PROCESSED_LOG"], 'w', newline='', encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)

            # เขียนแถวใหม่ต่อท้าย
            with open(cfg["PROCESSED_LOG"], 'a', newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    predict_id,
                    str(uuid.uuid4()),
                    filename,
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    cfg["JSON_RESULT_PREDICT_PERSON"],
                    cfg["JSON_RESULT_PREDICT_CLOTHING"],
                    save_name,
                    process_type,
                ])
        except Exception as e:
            print(f"[write_log 2] is error : {e}")
    def save_result_prediction(self,data):
        try:
            
            reFormData = data.dict()
            config = self.config

            for file in reFormData.files_name:
                self.write_log(file,config,reFormData.predict_id,'video_select_detection',reFormData.save_result_name)
            
            return {"message": "Custom detection saved successfully"}
        except Exception as e:
            print(f"[save_result_prediction] is error : {e}")

    def get_select_prediction_saved(self,post_data):
        try:

            path = post_data.clothing_detection_result_path
            if os.path.exists(path):
                data_list = self.load_json(path)
                load_data = []
                for filename, objects in data_list.items():
                    if objects[0]["predict_id"] == post_data.process_id:
                        load_data.append(objects)

                return load_data
            else:
                return []
        except Exception as e:
            print(f"[get_selcet_video_detection] is error : {e}")

    def get_data_saved(self,post_data):#ต้องแก้เกี่ยวกับสีใหม่
        try:

            path = post_data.CustomDetectionData.clothing_detection_result_path
            if os.path.exists(path):
                df = pd.read_csv(path)
            
                if post_data.custom_and_clolr_select:
                    filtered_results = []
                    for item in post_data.custom_and_clolr_select:
                        print(f'Processing class ID: {item.classId} with colors: {item.colors}')
                        filtered = df[df['class'] == item.classId].copy()
                        if not filtered.empty:
                            print(f'Found {len(filtered)} matches for class ID {item.classId}')
                            if item.colors:
                                try:
                                    filtered = self.color_manage.prepare_and_find_similar_colors(filtered, item.colors, 100)
                                    filtered_results.append(filtered)
                                except Exception as e:
                                    print(f'Error processing class {item.classId}: {str(e)}')
                                    continue
                            else:
                                filtered_results.append(filtered)

                    if filtered_results:
                        print(f'Found {len(filtered_results)} matches for class ID {item.classId}')
                        try:
                            filtered_results['process_id'] = post_data.CustomDetectionData.process_id
                            final_df = pd.concat(filtered_results, ignore_index=True)
                            return final_df.to_dict(orient="records")
                        except Exception as e:
                            print(f'Error concatenating results: {str(e)}')
                        # return []
                    # else:
                        # If no custom detection data or no matches, return all results
                        # return df.to_dict(orient="records")
                else:
                    return []
            else:
                return []
        except Exception as e:
           print(f"[get_data_saved] is error : {e}")