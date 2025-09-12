# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: E:\ALL_CODE\python\fashion-project\app\services\predict_service.py
# Bytecode version: 3.11a7e (3495)
# Source timestamp: 2025-09-07 02:38:15 UTC (1757212695)

import os
import cv2
import csv
import json
import datetime
from ultralytics import YOLO
import time
from app.services.data_services.data_service import DataManager
from app.services.config_service import get_config, save_config
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import uuid
from typing import Generator
from app.services.data_services.filtering_service import CSVSearchService
import statistics
import traceback
import sys

class Detection:
    def __init__(self):
        self.filter = CSVSearchService()
        self.data_manage = DataManager()
        self.config = get_config()
        self.CLASS_NAMES_B = ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress']
        self.CONFIG_PATH = 'resources/config.json'
        self.AI_MODEL_PATH = 'E:\\ALL_CODE\\python\\fashionV1\\fastapi-project\\model\\fashion13cls.pt'
        self.VIDEO_PATH = 'E:\\ALL_CODE\\nextjs\\fashion_project_1\\public\\videos'
        self.tracking_model = self.tracking_model = YOLO(self.config.get("AI_MODEL_PATH") + self.config.get("TRACKING_AI"))

    def predict_option_seting(self):

        dir = self.config['RESULTS_PREDICT_DIR']
        return self.get_result_csv(dir + 'clothing_detection', True, 'clothing_detection')
        


    def load_config_(self, custom_path=None):
        with open(self.CONFIG_PATH, 'r') as f:
            config = json.load(f)
            JSON_RESULT_PREDICT_PERSON = config['JSON_RESULT_PREDICT_PERSON']
            JSON_RESULT_PREDICT_CLOTHING = config['JSON_RESULT_PREDICT_CLOTHING']
            PROCESSED_LOG = config['PROCESSED_LOG']
            AI_MODEL_PATH = config['AI_MODEL_PATH'] + config['AI_MODEL_NAME']
            VIDEO_PATH = config['VIDEO_PATH']
            RESULTS_PREDICT_DIR = config['RESULTS_PREDICT_DIR']
            confidence = config['MODEL_CONFIG']['confidence_threshold']
            frequency = config['MODEL_CONFIG']['frequency']
            TRACKING_AI = config['AI_MODEL_PATH'] + config['TRACKING_AI'] 
            if custom_path:
                confidence = custom_path.confidence
                frequency = custom_path.frequency
                AI_MODEL_PATH = custom_path.custom_ai_path
                VIDEO_PATH = custom_path.video_path
                if custom_path.use_system_ai:
                    AI_MODEL_PATH = config['AI_MODEL_PATH'] + custom_path.system_model
                return {'JSON_RESULT_PREDICT_PERSON': JSON_RESULT_PREDICT_PERSON, 'JSON_RESULT_PREDICT_CLOTHING':  JSON_RESULT_PREDICT_CLOTHING, 'RESULTS_PREDICT_DIR': RESULTS_PREDICT_DIR, 'PROCESSED_LOG':  PROCESSED_LOG, 'AI_MODEL_PATH':  AI_MODEL_PATH, 'VIDEO_PATH':VIDEO_PATH, 'TRACKING_AI': TRACKING_AI, 'confidence': confidence, 'frequency': frequency}
            return {'JSON_RESULT_PREDICT_PERSON': JSON_RESULT_PREDICT_PERSON, 'JSON_RESULT_PREDICT_CLOTHING': JSON_RESULT_PREDICT_CLOTHING, 'PROCESSED_LOG':  PROCESSED_LOG, 'AI_MODEL_PATH':  AI_MODEL_PATH, 'VIDEO_PATH': VIDEO_PATH, 'RESULTS_PREDICT_DIR':RESULTS_PREDICT_DIR, 'TRACKING_AI':  TRACKING_AI, 'confidence': confidence, 'frequency': frequency}

    def get_result_csv(self, dir, detect_all, type_of_detection):
        type_of_detection = datetime.datetime.now().strftime('%Y%m%d')
        os.makedirs(dir, exist_ok=True)
        if detect_all:
            base_name = os.path.join(dir, f'results_{type_of_detection}_{type_of_detection}.json')
            if not os.path.exists(base_name):
                return base_name
            counter = 1
            while True:
                new_name = os.path.join(dir, f'results_{type_of_detection}_{type_of_detection}_{counter}.json')
                if not os.path.exists(new_name):
                    return new_name
                counter = counter + 1
        files = [f for f in os.listdir(dir) if f.startswith(f'results_{type_of_detection}_{date_str}')]
        if files:
            return os.path.join(dir, sorted(files)[(-1)])
        return os.path.join(dir, f'results_{type_of_detection}_{type_of_detection}.json')

    def load_processed_files(self):
        config = self.load_config_()
        if not os.path.exists(config.get('PROCESSED_LOG')):
            return set()
        with open(config.get('PROCESSED_LOG'), 'r') as f:
            return set((line.strip() for line in f))

    def mark_file_as_processed(self, filename):
        with open(self.config.get('PROCESSED_LOG'), 'a') as f:
            f.write(filename + '\n')

    def write_log(self, filename, cfg, predict_id, process_type):
        os.makedirs(os.path.dirname(cfg['PROCESSED_LOG']), exist_ok=True) if os.path.dirname(cfg['PROCESSED_LOG']) else None
        headers = ['predict_id', 'process_id', 'filename', 'datetime', 'person_detection_result_path', 'clothing_detection_result_path', 'save_result_name', 'process_type']
        if not os.path.exists(cfg['PROCESSED_LOG']):
            with open(cfg['PROCESSED_LOG'], 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
        with open(cfg['PROCESSED_LOG'], 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([predict_id, str(uuid.uuid4()), filename, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cfg['JSON_RESULT_PREDICT_PERSON'], cfg['JSON_RESULT_PREDICT_CLOTHING'], 'detect', process_type])

    def get_video_files(self, folders):
        video_files = []
        for folder in folders:
            print(folder)
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov')):
                        video_files.append(os.path.join(root, file))
        print(f'[get_video_files] พบ {len(video_files)} ไฟล์')
        return video_files

    def clean_data(self, detections):
        df = pd.DataFrame(detections)
        if df.empty:
            return []
        top_classes = df['class'].value_counts().head(2).index.tolist()
        df_top = df[df['class'].isin(top_classes)]
        best_per_class = df_top.loc[df_top.groupby('class')['confidence'].idxmax()].to_dict(orient='records')
        return best_per_class

    def find_clothing(self, data):

        try:
            gruop_data = pd.DataFrame(data).groupby('track_id')
            body_T_class = [0, 1, 2, 3, 4, 5]
            body_B_class = [6, 7, 8]
            all_body_class = [9, 10, 11, 12]
            return_data = pd.DataFrame()
            for id, df in gruop_data:
                top_classes = df['class'].value_counts().head(2).index.tolist()
                try:
                    top1, top2 = top_classes
                except:
                    return_data = pd.concat([df, return_data], ignore_index=True)
            
                if (top1 in all_body_class and top2 in all_body_class or (top1 in body_B_class or top2 in body_B_class)) and (top1 in all_body_class or top2 in all_body_class):
                    save_top1 = top1
                    save_top2 = top2
                    if save_top1 not in all_body_class:
                        top1 = save_top2
                        top2 = save_top1
                    mask = df['class'].isin(all_body_class)
                    df.loc[mask, 'class'] = top1
                    false_list = df[~mask]
                    track_group = false_list.groupby('track_id')
                    new_rows = []
                    for track_id, objs in track_group:
                        frame_group = objs.groupby('timestamp')
                        for timeframe, frames in frame_group:
                            row = frames.iloc[0]
                            x_clothing, y_clothing, w_clothing, h_clothing = ([], [], [], [])
                            if row['class'] in body_T_class:
                                x_clothing = row['x_clothing']
                                y_clothing = row['y_clothing']
                                w_clothing = row['w_clothing']
                                h_clothing = 2  * row['h_clothing']
                            else:  # inserted
                                if row['class'] in body_B_class:
                                    x_clothing = row['x_clothing']
                                    y_clothing = row['y_clothing']+row['h_clothing']
                                    w_clothing = row['w_clothing']
                                    h_clothing = 2* row['h_clothing']
                            new_data = {**row.to_dict(), 
                                        'timestamp': timeframe, 
                                        'class': top1, 
                                        'class_name': self.CLASS_NAMES_B[top1] if top1 < len(self.CLASS_NAMES_B) else str(top1), 
                                        'x_clothing': x_clothing, 
                                        'y_clothing': y_clothing, 
                                        'w_clothing': w_clothing, 
                                        'h_clothing': h_clothing}
                            new_rows.append(new_data)
                    new_df = pd.DataFrame(new_rows)
                    return_data = pd.concat([df, new_df, return_data], ignore_index=True)

                
                elif top1 in body_T_class and top2 in body_B_class:
                    df.loc[df['class'].isin(body_T_class), 'class'] = top1
                    df.loc[df['class'].isin(body_B_class), 'class'] = top2
                    mask = df['class'].isin(all_body_class)
                    true_list = df[mask]
                    track_group = true_list.groupby('track_id')
                    new_rows = []
                    for track_id, objs in track_group:
                        frame_group = objs.groupby('timestamp')
                        for timeframe, frames in frame_group:
                            row = frames.iloc[0]
                            new_data1 = {
                                            **row.to_dict(),
                                            'timestamp':timeframe , 
                                            'class': top1, 
                                            'class_name':  self.CLASS_NAMES_B[top1] if top1 < len(self.CLASS_NAMES_B) else str(top1), 
                                            'h_clothing': row['h_clothing'] / 2,
                                        }
                            new_data2 = {
                                            **row.to_dict(),
                                            'timestamp': timeframe,
                                            'class': top2,
                                            'class_name': self.CLASS_NAMES_B[top2] if top2 < len(self.CLASS_NAMES_B) else str(top2),
                                            'y_clothing': int(row['y_clothing']) - int(row['h_clothing']) /2,
                                            'h_clothing': int(row['h_clothing']) /2
                                        }                       
                            new_rows.append(new_data1)
                            new_rows.append(new_data2)
                    new_df = pd.DataFrame(new_rows)
                    return_data = pd.concat([df, new_df, return_data], ignore_index=True)

                elif top2 in body_T_class and top1 in body_B_class:
                    df.loc[df['class'].isin(body_T_class), 'class'] = top2
                    df.loc[df['class'].isin(body_B_class), 'class'] = top1
                    mask = df['class'].isin(all_body_class)
                    true_list = df[mask]
                    track_group = true_list.groupby('track_id')
                    new_rows = []
                    for track_id, objs in track_group:
                        frame_group = objs.groupby('timestamp')
                        for timeframe, frames in frame_group:
                            row = frames.iloc[0]
                            new_data1 = {
                                            **row.to_dict(),
                                            'timestamp': timeframe,
                                            'class': top2,
                                            'class_name': self.CLASS_NAMES_B[top2] if top2 < len(self.CLASS_NAMES_B) else str(top2),
                                            'h_clothing': int(row['h_clothing']) /2
                                        }

                            new_data2 = {
                                            **row.to_dict(),
                                            'timestamp': timeframe,
                                            'class': top1,
                                            'class_name': self.CLASS_NAMES_B[top1] if top1 < len(self.CLASS_NAMES_B) else str(top1),
                                            'y_clothing': int(row['y_clothing']) - int(row['h_clothing']) / 2,
                                            'h_clothing': int(row['h_clothing']) / 2
                                        }
                            new_rows.append(new_data1)
                            new_rows.append(new_data2)
                    new_df = pd.DataFrame(new_rows)
                    return_data = pd.concat([df, new_df, return_data], ignore_index=True)


                elif (top1 in body_T_class or top2 in body_T_class) and (top1 in all_body_class or top2 in all_body_class):
                    mask = df['class'].isin(body_B_class)
                    false_list = df[~mask]
                    true_list = df[mask]
                    class_id_top = top1 if top1 in body_T_class else top2
                    class_id_dress = top2 if top1 in body_T_class else top1
                    false_list.loc[false_list['class'].isin(body_T_class), 'class'] = class_id_top
                    false_list.loc[false_list['class'].isin(all_body_class), 'class'] = class_id_dress
                    df = pd.concat([false_list, true_list], ignore_index=True)
                    track_group = true_list.groupby('track_id')
                    new_rows = []
                    for track_id, objs in track_group:
                        frame_group = objs.groupby('timestamp')
                        for timeframe, frames in frame_group:
                            row = frames.iloc[0]
                            new_data1 = {
                                **row.to_dict(),
                                'timestamp': timeframe,
                                'class': class_id_top,
                                'class_name': self.CLASS_NAMES_B[class_id_top] if class_id_top < len(self.CLASS_NAMES_B) else str(class_id_top),
                                'y_clothing': int(row['y_clothing']) - int(row['h_clothing'])  # ✅ รวม y + h
                            }

                            new_data2 = {
                                **row.to_dict(),
                                'timestamp': timeframe,
                                'class': class_id_dress,
                                'class_name': self.CLASS_NAMES_B[class_id_dress] if class_id_dress < len(self.CLASS_NAMES_B) else str(class_id_dress),
                                'y_clothing': int(row['y_clothing']) - int(row['h_clothing']),
                                'h_clothing': int(row['h_clothing']) * 2  # ✅ ขยายกรอบขึ้น 2
                            }
                            new_rows.append(new_data1)
                            new_rows.append(new_data2)
                    new_df = pd.DataFrame(new_rows)
                    return_data = pd.concat([df, new_df, return_data], ignore_index=True)
                else:
                    return_data = pd.concat([df, return_data], ignore_index=True)
                    
            return return_data

        except Exception as e:
                    print(f'\033[91m[find_clothing]\033[0m is error : {e}')
                    traceback.print_exc()
                    return df

    def show_color_window(self, color_bgr, name='Color'):
        swatch = np.zeros((100, 100, 3), dtype=np.uint8)
        swatch[:] = np.uint8(color_bgr)
        cv2.imshow(name, swatch)

    def get_color1(self, model, image, width, height):
        result = model.predict(source=image, verbose=False)[0]
        mean_color_bgr = [0, 0, 0]
        if result.masks is not None and len(result.masks.data) > 0:
            mask = result.masks.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_resized.astype(bool)
            pixels_in_mask = image[mask_bool]
            mean_color_bgr = pixels_in_mask.mean(axis=0)
        return mean_color_bgr

    def get_color_dominant(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape((-1), 3)
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        counts = np.bincount(kmeans.labels_)
        dominant = colors[np.argmax(counts)][::(-1)]
        return str(dominant)

    def get_color_percentage_with_threshold(self, image, threshold=200):
        """   แยก object จาก background ด้วย threshold
        image: BGR image
        threshold: ค่าความสว่างเพื่อแยก background (0-255)\n    """  # inserted
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, object_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        total_object_pixels = cv2.countNonZero(object_mask)
        height, width = image.shape[:2]
        total_pixels = height * width  # จำนวน pixel ทั้งหมด
        total_background_pixels = total_pixels - total_object_pixels  # pixel ที่เป็นพื้นหลัง
        color_ranges = {
            'Red': [(0, 50, 40), (10, 255, 255)],
            'Orange': [(11, 50, 128), (25, 255, 255)],
            'Yellow': [(26, 50, 40), (35, 255, 255)],
            'LightGreen': [(36, 50, 40), (60, 255, 255)],
            'Green': [(61, 50, 40), (85, 255, 255)],
            'Cyan': [(86, 50, 40), (100, 255, 255)],
            'Blue': [(101, 50, 40), (135, 255, 255)],
            'Violet': [(136, 50, 40), (160, 255, 255)],
            'Pink': [(161, 30, 150), (170, 255, 255)],
            'Magenta': [(171, 50, 50), (180, 255, 255)],
            'White': [(0, 0, 200), (180, 30, 255)],
            'Black': [(0, 0, 0), (180, 255, 50)],
            'Brown': [(10, 150, 50), (20, 255, 150)]
        }
        percentages_object = {}
        percentages_background = {}
        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            object_count = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=object_mask))
            percentages_object[color] = total_object_pixels + 0 + (object_count + total_object_pixels) + 100 if total_object_pixels > 0 else 0
            background_mask = cv2.bitwise_not(object_mask)
            background_count = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=background_mask))
            percentages_background[color] = 100 if total_background_pixels > 0 else 0
        if 'Red' in percentages_object and 'Red2' in percentages_object:
            percentages_object['Red'] = percentages_object.pop('Red2')
            percentages_background['Red'] = percentages_background.pop('Red2')
        return (percentages_object, percentages_background)

    def detect_objects(self, video_path, model_A, model_B, output_csv, cfg, result_people_detection_csv, filename, class_selected=None):
        try:

            cap = cv2.VideoCapture(video_path)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_index = 0
            result_detection_clothing = []
            people_detections = [] 

            seen_track_ids = set()
            frame_interval = int(fps * cfg['frequency'])
            filename = os.path.basename(video_path)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                else:  # inserted
                    
                    if (frame_index % frame_interval) == 0:
                        results = model_A.track(source=frame, classes=[0], conf=0.5, iou=0.5, tracker='bytetrack.yaml', verbose=False, show=False, save=False)
                        if results[0].boxes is not None and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                track_id = int(box.id.item()) if box.id is not None else (-1)
                                person_crop = frame[y1:y2, x1:x2]
                                timestamp = frame_index / fps
                                people_detections.append({'predict_id': str(uuid.uuid4()), 
                                                        'filename': filename, 
                                                        'timestamp': timestamp, 
                                                        'width': width, 
                                                        'height': height, 
                                                        'track_id': track_id, 
                                                        'x_person': x1, 
                                                        'y_person': y1, 
                                                        'w_person': x2 - x1, 
                                                        'h_person': y2 - y1})
                                conf_thres = cfg.get('confidence')
                                predict_results = model_B.predict(source=person_crop, verbose=False, conf=conf_thres, iou=0.5, classes=None if not class_selected else class_selected)[0]
                                for idx, pbox in enumerate(predict_results.boxes):
                                    cls_b = int(pbox.cls[0])
                                    conf_b = float(pbox.conf[0])
                                    x1_clothing, y1_clothing, x2_clothing, y2_clothing = map(int, pbox.xyxy[0])
                                    crop_clothing = person_crop[y1_clothing:y2_clothing, x1_clothing:x2_clothing]
                                    obj_pct, bg_pct = self.get_color_percentage_with_threshold(crop_clothing, threshold=200)
                                    result_detection_clothing.append({'predict_id': str(uuid.uuid4()), 
                                                                    'filename': filename, 
                                                                    'width': width, 
                                                                    'height': height, 
                                                                    'timestamp': timestamp, 
                                                                    'class': int(cls_b), 
                                                                    'class_name': self.CLASS_NAMES_B[cls_b], 
                                                                    'confidence': round(conf_b, 2), 
                                                                    'x_person': x1, 
                                                                    'y_person': y1, 
                                                                    'w_person': x2 - x1, 
                                                                    'h_person': y2 - y1, 
                                                                    'x_clothing': x1_clothing, 
                                                                    'y_clothing': y1_clothing, 
                                                                    'w_clothing': x2_clothing - x1_clothing, 
                                                                    'h_clothing': y2_clothing - y1_clothing, 
                                                                    'track_id': track_id, 
                                                                    'colors': obj_pct})
                                progress = (frame_index / total_frames) * 100
                                yield {'progress': round(progress, 2) }

                    frame_index = frame_index + 1
            print("befor",len(result_detection_clothing))
            tuned_data = self.find_clothing(result_detection_clothing)
            print("after",len(tuned_data)) 
            detection_clothing_tuned = tuned_data.to_dict(orient='records')
            cap.release()
            self.data_manage.update_result_to_json(result_people_detection_csv, people_detections)
            self.data_manage.update_result_to_json(output_csv, detection_clothing_tuned)
            new_path = {}
            new_path.update({'JSON_RESULT_PREDICT_PERSON': result_people_detection_csv})
            new_path.update({'JSON_RESULT_PREDICT_CLOTHING': output_csv})
            save_config(new_path)
            return detection_clothing_tuned
        except Exception as e:
            print(f'[91m[detect_objects][0m is error : {e}')
            # traceback.print_exc()

    def predict_img_clothing(self,filename, bbox_xyxy, image , track_id ,class_selected=None) -> Generator[dict, None, None]:
        """Predict clothing จาก video และ return progress ทีละ frame"""  # inserted
        



        config = self.load_config_()
        conf_thres = config['confidence']
        clean_detections_clothing = []
        x1,y1
        for i, box in enumerate(bbox_xyxy):
                xp1, yp1, xp2, yp2 = [int(i) for i in box]
                cap = image[yp1:yp2, xp1:xp2]
                predict_results = self.tracking_model.predict(source=cap, verbose=False, conf=conf_thres, iou=0.5, classes=None if not class_selected else class_selected)[0]
                detections = []
                for pbox in predict_results.boxes:
                    cls_b = int(pbox.cls[0])
                    conf_b = float(pbox.conf[0])
                    x1, y1, x2, y2 = map(int, pbox.xyxy[0])
                    crop_clothing = cap[y1:y2, x1:x2]
                    obj_pct, bg_pct = self.get_color_percentage_with_threshold(crop_clothing, threshold=200)
                    detections.append({ 'predict_id': str(uuid.uuid4()), 
                                        'filename': filename, 
                                        'timestamp': "", 
                                        'class': cls_b, 
                                        'class_name': self.CLASS_NAMES_B[cls_b] if cls_b < len(self.CLASS_NAMES_B) else str(cls_b), 
                                        'confidence': round(conf_b, 2),
                                        'x': x1, 
                                        'y': y1,
                                        'w': x2 - x1, 
                                        'h': y2 - y1, 
                                        'track_id':track_id,
                                        'mean_color_bgr': obj_pct
                                    })
                clean_detections_clothing.extend(detections)
                # yield {'frame': frame_index, 'progress': round(frame_index + cap.get(cv2.CAP_PROP_FRAME_COUNT) + 100, 2), 'detections': detections}
        return clean_detections_clothing
        

    def save_result(self, output_csv, fieldnames, result):
        file_exists = os.path.exists(output_csv)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in result:
                writer.writerow(row)

    def process_videos(self, detect_all=True, custom_config=None):
        try:
            cfg = self.load_config_(custom_config)
            model_A = YOLO(cfg['TRACKING_AI']).to('cuda')
            model_B = YOLO(cfg['AI_MODEL_PATH']).to('cuda')
            print('start detect_objects')
            print('model A: ', cfg['TRACKING_AI'])
            print('model B: ', cfg['AI_MODEL_PATH'])
            dir = cfg['RESULTS_PREDICT_DIR']

            output_ = self.get_result_csv(dir + 'clothing_detection', detect_all, 'clothing_detection')
            result_people_detection_csv = self.get_result_csv(dir + 'people_detection', detect_all, 'people_detection')
            video_files = self.get_video_files([cfg['VIDEO_PATH']])
            if len(video_files) == 0:
                print('no video')
                return
            predict_id = str(uuid.uuid4())
            for video in video_files:
                
                start_track = time.perf_counter()
                filename = os.path.basename(video)
                print(f'[▶] Processing: {video}')
                yield {'status': 'start', 'video': video}
                yield from self.detect_objects(video, model_A, model_B, output_, cfg, result_people_detection_csv, filename, class_selected=None)
                self.write_log(filename, cfg, predict_id, 'normal_detection')
                track_time = time.perf_counter() - start_track
                print(track_time)
                yield {'status': 'done', 'video': video}
            print('predict success!!')
        except Exception as e:
            print(f'[process_videos] is error : {e}')
            # traceback.print_exc()

    def upload_video(self,video_file):
        try:
            video_path = os.path.join(self.config.get('UPLOAD_VIDEO_PATH', ''), video_file.filename)
            with open(video_path, 'wb') as f:
                f.write(video_file.file.read())
                return video_path
        except Exception as e:
                print(f'Error uploading video[upload_video]: {e}')
                return None

    def start_process_select_detection(self, model, custom_detection_data, files):
        try:
            model_A = YOLO(self.config.get('AI_MODEL_PATH', '') + self.config.get('TRACKING_AI', '')).to('cuda')
            model_B = YOLO(self.config.get('AI_MODEL_PATH', '') + model).to('cuda')
            dir = self.config.get('CUSTOM_RESULT_PREDICT_DIR', '')
            output_csv = self.get_result_csv(dir + 'clothing_detection', True, 'clothing_detection')
            resultpeople_detection_csv = self.get_result_csv(dir + 'people_detection', True, 'people_detection')
            class_selected = []
            result = []
            cfg = self.load_config_()
            if custom_detection_data:
                for class_ in json.loads(custom_detection_data):
                    class_selected.append(class_['classId'])
            predict_id = str(uuid.uuid4())
            files_name = []
            for video in files:
                video_path =self.upload_video(video)
                filename = video.filename
                print(f'[▶] Processing: {filename}')
                files_name.append(filename)
                if not class_selected:
                    detect = self.detect_objects(video_path, model_A, model_B, output_csv, cfg, resultpeople_detection_csv)
                else:  # inserted
                    detect = self.detect_objects(video_path, model_A, model_B, output_csv, cfg, resultpeople_detection_csv, class_selected)
                result.extend(detect)
            df = pd.DataFrame(result)
            filtered_results = []
            if custom_detection_data:
                for class_ in json.loads(custom_detection_data):
                    print(f"Processing class ID: {class_['classId']} with colors: {class_['colors']}")
                    filtered = df[df['class'] == class_['classId']].copy()
                    print('filtered', filtered)
                    if not filtered.empty:
                        print(f"Found {len(filtered)} matches for class ID {class_['classId']}")
                        filtered = filter.prepare_and_find_similar_colors(filtered, class_['colors'], 100)
                        filtered_results.append(filtered)
                        print(f"Error processing class {class_['classId']}: {str(e)}")
            predict_id = str(uuid.uuid4())
            if filtered_results:
                    print(f"Found {len(filtered_results)} matches for class ID {class_['classId']}")
                    final_df = pd.concat(filtered_results, ignore_index=True)
                    result = {'data': final_df.to_dict(orient='records'), 'person_detection_result_path': resultpeople_detection_csv, 'clothing_detection_result_path': output_csv, 'predict_id': predict_id, 'files_name': files_name}
                    return result
            else:
                result = {'data': df.to_dict(orient='records'), 'person_detection_result_path': resultpeople_detection_csv, 'clothing_detection_result_path': output_csv, 'predict_id': predict_id, 'files_name': files_name}
                return result
        except Exception as e:
                    print(f'[start_process_select_detection] is error : {e}')
                    return None



