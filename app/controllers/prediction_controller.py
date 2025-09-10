from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from app.services.predict_service import process_videos,start_process_select_detection
from app.schemas.data_schema import setupPredice
from typing import List
router = APIRouter()


@router.post("/predict-clothing-scaning")
def predict_clothing_api(data: setupPredice):
    try:

        if data.auto_config:
     
            result = process_videos(detect_all=data.detectingAll)
        else:

            result = process_videos(detect_all=data.detectingAll, custom_config=data)

        return StreamingResponse(
            (f"{item}\n" for item in result),
            media_type="application/json"
    )
    except Exception as e:
        print(f"[processing] Error: {e}")

@router.post("/custom-detection-start")
async def api_custom_detection_start(
    model: str = Form(...),
    # date_range: str = Form(...),
    custom_detection_data: str = Form(...),
    files: List[UploadFile] = File(...)
):
    return start_process_select_detection(model,custom_detection_data,files)

@router.post("/predict-pic") #แก้เพิ่มอีกเยอะ เรื่องสี
async def api_predict_picture(file: UploadFile = File(...)):
    return 