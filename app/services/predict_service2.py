from ultralytics import YOLO
from app.services.color_service import ColorCheck
from app.services.config_service import load_config
from app.services.data_services.data_service import DataManager
from lib.deepsort.tracking import run
import uuid
import time
from pathlib import Path
import datetime
import csv
import pandas as pd
import numpy as np
import cv2
from mss import mss
from PIL import Image
import traceback


manager = DataManager()
colorC = ColorCheck()
config = load_config()
model_pred_clothing = YOLO(config.get("AI_MODEL_PATH")+config.get("AI_MODEL_NAME"))
main_model_path =config.get("AI_MODEL_PATH")+'yolo11m.pt'

data = r"E:\ALL_CODE\python\fashion-project\lib\deepsort\data\coco.yaml"


def prediction(xyxy,identities,frame):
    try:    
        xp1,yp1,xp2,yp2 = xyxy
        crop = frame[yp1:yp2, xp1:xp2]
        results= model_pred_clothing.predict(crop)[0]
        clothing_list=[]
        object_data = { 
                        'x_person': xp1, 
                        'y_person': yp1,
                        'w_person': xp2 - xp1, 
                        'h_person': yp2 - yp1, 
                        'track_id':identities,
                    }
        if results:
            boxes = results.boxes
            for box in boxes:
                xc1,yc1,xc2,yc2 = map(int, box.xyxy[0].cpu().numpy())
                cls = int(box.cls.cpu().item())        # class index เป็น int
                conf = float(box.conf.cpu().item())    # confidence เป็น float
                crop_clothing = crop[yc1:yc2 ,xc1:xc2]
                crop_clothing = np.ascontiguousarray(crop_clothing)
                list_color = colorC.get_color_percentage_with_threshold(crop_clothing)
                clothing_list.append({ 
                                        **object_data,
                                        'predict_id': str(uuid.uuid4()), 
                                        'class': cls, 
                                        'class_name': model_pred_clothing.names[cls], 
                                        'confidence': round(conf, 2), 
                                        'x_clothing': xc1,
                                        'y_clothing': yc1,
                                        'w_clothing': xc2 - xc1,
                                        'h_clothing': yc2 - yc1,
                                        'mean_color_bgr': list_color
                                    })
        else:
            clothing_list.append({ 
                                        **object_data,
                                        'predict_id': str(uuid.uuid4()), 
                                        'class': 'undifined', 
                                        'class_name': 'undifined', 
                                        'confidence': 'undifined', 
                                        'x_clothing': 0,
                                        'y_clothing': 0,
                                        'w_clothing': 0,
                                        'h_clothing': 0,
                                        'mean_color_bgr': []
                                    })

        return clothing_list
    except Exception as e:
        print(f'\033[91m[detection]\033[0m is error : {e}')
        traceback.print_exc()

def get_result_csv(dir, detect_all, type_of_detection):
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        output_dir = Path(dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if detect_all:
            base_name = output_dir / f"results_{type_of_detection}_{date_str}_1.json"
            if not Path(base_name).exists():
                return str(base_name)
            counter = 2
            while True:
                new_name = output_dir /  f'results_{type_of_detection}_{date_str}_{counter}.json'
                if not Path(new_name).exists():
                    return str(new_name)
                counter = counter + 1
        else:
            files = [f for f in [p.name for p in output_dir.iterdir()] if f.startswith(f'results_{type_of_detection}_{date_str}')]
            if files:
                return output_dir / sorted(files)[(-1)]
        return str(output_dir / f'results_{type_of_detection}_{type_of_detection}.json')

def get_video_files( folders):
    video_files = []
    for folder in folders:
        print(folder)
        for file in Path(folder).rglob("*"):
            if file.suffix.lower() in (".mp4", ".avi", ".mov"):
                video_files.append(str(file))
    print(f"[get_video_files] พบ {len(video_files)} ไฟล์")
    return video_files


def write_log( filename, process_id, process_type):
    try:
        log_path = Path(config['PROCESSED_LOG'])
        
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        if log_path.parent:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        headers = [
            'predict_id', 'process_id', 'filename', 'datetime',
            'person_detection_result_path', 'clothing_detection_result_path',
            'save_result_name', 'process_type'
        ]

        # เขียน header ถ้าไฟล์ยังไม่ถูกสร้าง
        if not log_path.exists():
            with log_path.open('w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)

        # เขียน log ใหม่ (append)
        with log_path.open('a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                process_id,
                str(uuid.uuid4()),
                filename,
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                config['JSON_RESULT_PREDICT_PERSON'],
                config['JSON_RESULT_PREDICT_CLOTHING'],
                'detect',
                process_type
            ])
    except Exception as e:
        print(f'\033[91m[write_log]\033[0m is error : {e}')

def get_wh(source):
    """
    Return (width, height) of the source.
    Supports image, video, webcam (numeric), URL stream, screenshot.
    """
    IMG_FORMATS = ['jpg','jpeg','png','bmp','tif','tiff','dng','webp','mpo']
    VID_FORMATS = ['mp4','mov','avi','mkv','wmv','flv','mpg','mpeg','ts']
    source = str(source)
    is_file = Path(source).suffix[1:].lower() in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://','rtmp://','http://','https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')

    if is_file:  # Image or video file
        suffix = Path(source).suffix[1:].lower()
        if suffix in IMG_FORMATS:  # Image
            img = Image.open(source)
            w, h = img.size
        else:  # Video
            cap = cv2.VideoCapture(source)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

    elif webcam:  # Webcam or stream
        cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    elif screenshot:  # Screen capture
        with mss() as sct:
            monitor = sct.monitors[0]  # full screen
            w = monitor['width']
            h = monitor['height']

    elif is_url:  # URL stream (not a file)
        cap = cv2.VideoCapture(source)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    else:
        raise ValueError(f"Unknown source type: {source}")

    return w, h

def predict_ctrl(data_batch,filename,source):
    try:

        all_result = []
        h, w = get_wh(source)
        i=0
        for bbox_xyxy, identities, ims in data_batch:
            i+=1
            for i, box in enumerate(bbox_xyxy):
                all_result.extend(prediction(box,identities[i],ims))
        if all_result:
            df = pd.DataFrame(all_result)
            df['filename'] = filename
            df['w_vid'] = w
            df['h_vid'] = h
            return df

    except Exception as e:
        print(f'\033[95m[predict_ctrl]\033[0m is error : {e}')


def processing_videos():
        
        try:
            process_status:bool = False
            process_id = str(uuid.uuid4())
            dir = config.get('RESULTS_PREDICT_DIR')
            path_to_save = get_result_csv(dir + 'clothing_detection', True, 'clothing_detection')
            print('see result at :\033[93m',path_to_save,'\033[0m')
            video_files = get_video_files([config['VIDEO_PATH']])
            if len(video_files) == 0:
                print('no video')
                return
            
            for video in [video_files[1]]:
                
                start_track = time.perf_counter()
                filename = Path(video).name
                print(f'[▶] Processing: {video}')

                predict = run(weights=main_model_path,source= video,data=data,classes=0,pred_clothing=True)

                if not predict.empty:
                    num_rows = predict.shape[0]
                    print('predicted data ',num_rows)
                    predict['process_id'] = process_id
                    manager.update_result_to_json(path_to_save,predict.to_dict(orient='records'))
                    write_log(filename, process_id, 'normal_detection')
                    track_time = time.perf_counter() - start_track
                    print(track_time)
                    process_status=True
            if process_status:
                print('predict success!!')
            else:
                print('process detection fail!!')
        except Exception as e:
            print(f'[processing_videos] is error : {e}')
            traceback.print_exc()

if __name__ == 'main':
    processing_videos()