# from app.services.predict_and_tracking import run_predict
from app.services.predict_service2 import processing_videos

def predict():
    # deepsort
    # predict + tracking + predict clothing 
    processing_videos()

if __name__ == "__main__":
    predict()