from app.services.config_service import load_config
import os


config = load_config()

def scan_folder(folder_path,target_extensions):
    try:

        file_list = [
            entry.name
            for entry in os.scandir(folder_path)
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in target_extensions
        ]

        # print(f"เจอทั้งหมด {len(file_list)} ไฟล์")
        return file_list 
    except Exception as e: 
        print(f"Error: {e}")

def scan_ai_model():
    folder_path = config.get("AI_MODEL_PATH", "")
    target_extensions = {".pt"}  # ใส่ชนิดไฟล์ที่ต้องการ
    return scan_folder(folder_path,target_extensions)

       

def scan_video_floder():
    folder_path = config.get("VIDEO_PATH", "")
    target_extensions = {".mp4", ".avi", ".mov"}  # ใส่ชนิดไฟล์ที่ต้องการ
    return scan_folder(folder_path,target_extensions)

def check_video_unscanned():
    predicted_video = config.get("PROCESSED_LOG", "")
    with open(predicted_video, 'r') as f:
        predicted_video_name = f.read()
        current_video = scan_video_floder()
        yet_predicted = list(set(current_video) - set(predicted_video_name.split("\n")))
        return yet_predicted

def scan_all():
    current_path=config.get("VIDEO_PATH", "")
    model_ai=scan_ai_model()
    video_floder=scan_video_floder()
    video_unscanned=check_video_unscanned()
    predicted_file_list=list(set(video_floder) - set(video_unscanned))
    # default config
    confidence_threshole=config.get("MODEL_CONFIG")["confidence_threshold"]
    frequency=config.get("MODEL_CONFIG")["frequency"]
    detectingAll=config.get("MODEL_CONFIG")["detectingAll"]
    return {"current_path":current_path,
            "available_models":model_ai,
            "total_files":len(video_floder),
            "unpredicted_files":len(video_unscanned),
            "predicted_files":len(predicted_file_list),
            "unpredicted_file_list":video_unscanned,
            "confidence_threshole":confidence_threshole,
            "frequency":frequency,
            "detectingAll":detectingAll}