from pydantic import BaseModel

class preData(BaseModel):
    filename: list[str]
    class_color: list[dict]
    class_collab: bool
    date: str = None
    color_collab: bool


class setupPredice(BaseModel):
    auto_config: bool
    video_path: str
    use_system_ai: bool
    system_model: str
    custom_ai_path: str
    confidence: float
    frequency: float
    detectingAll: bool

class resultVideoDetail(BaseModel):
    filename: str
    process_id: str

class pathToResult(BaseModel):
    path: str
    file_name: str

class AnalyzeVideo(BaseModel):
    file_name: str
    file_path: str
    predictionStrategyClothing: str
    predictionStrategyPerson: str

class CustomDetectionDataSave(BaseModel):
    person_detection_result_path: str
    clothing_detection_result_path: str
    save_result_name: str
    predict_id: str
    files_name: list[str]   

class SelectDetectionDataGet(BaseModel):
    person_detection_result_path: str
    clothing_detection_result_path: str
    save_result_name: str
    predict_id: str

class classAndColor(BaseModel):
    classId: int
    colors: list[str]

class GetDetectionSelectData(BaseModel):
    CustomDetectionData:SelectDetectionDataGet
    custom_and_clolr_select: list[classAndColor] 