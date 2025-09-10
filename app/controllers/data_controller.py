from fastapi import APIRouter 
from services.data_services.filtering_service import CSVSearchService
from services.data_services.data_service import DataManager
from schemas.data_schema import preData,resultVideoDetail,pathToResult,AnalyzeVideo,CustomDetectionDataSave,SelectDetectionDataGet,GetDetectionSelectData
from services.data_services.process_check_service import scan_all
from services.config_service import get_config

config = get_config()

router = APIRouter()
searcher = CSVSearchService()
data_manage = DataManager()


@router.post("/pre-data")
def api_get_all_video_name(filter:preData):
    return  searcher.filter_data(filter)

@router.get("/process-check")
def api_process_check():
    return scan_all()

@router.get("/get-all-resault")
def api_get_all_result():
    return data_manage.load_json(config.get("JSON_RESULT_PREDICT_CLOTHING",""))

@router.post("/video-detail")
def api_get_video_detail(data:resultVideoDetail):
    return data_manage.get_result_detail(data)

@router.get("/get-all-video")
def api_get_all_video_():
    return data_manage.get_all_video_name() 

@router.get("/get_model_name")
def api_get_ai_model_name():
    return data_manage.get_dot_pt_file()
@router.post("/get-result-predicted")
def api_get_result_by_path(data:pathToResult):
    """get data จาก path ที่กำหนด""" 
    return data_manage.load_json(data.path+'/'+data.file_name) 

@router.post("/analyze-video")
def api_analyze_video(data:AnalyzeVideo):
    return data_manage.analyze_result_predict_data(data) 

@router.get("/get-log-detection")
def api_get_log_select_detection():
    return data_manage.get_log_selcet_video_detection()

@router.post("/save-result-prediction")
def api_save_result_prediction(data:CustomDetectionDataSave):
    return data_manage.save_result_prediction(data)
    
@router.post("/get-select-prediction-saved")
def api_load_select_prediction_saved(data:SelectDetectionDataGet):
    return data_manage.get_select_prediction_saved(data)

@router.post("/get-saved-data")
def api_get_data_saved(data:GetDetectionSelectData):
    return 

