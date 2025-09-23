import argparse

from app.services.tune_service import process
from app.services.predict_and_tracking import run_predict
from app.services.predict_service2 import processing_videos

def predict():
    # deepsort
    # predict + tracking + predict clothing 
    processing_videos()

def main():
    # สร้าง parser
    parser = argparse.ArgumentParser(description="ตัวอย่าง argparse")
     # เพิ่ม argument
    parser.add_argument("--start", default=True, type=str, help="")
    parser.add_argument("--mode", type=int, help="predict, check, tune")
    parser.add_argument("--file", default=None, type=str, help="ที่อยู่ file video ถ้า mode คือ predict use result.json path for mode tune")


    # parse argument จาก command line
    args = parser.parse_args()
    file = args.file
    if args.start:
        if args.mode == 'predict':
            run_predict()
        elif args.mode == 'check':
            processing_videos()
        elif args.mode == 'tune':
            process(path=file)
        else:
            processing_videos()

if __name__ == "__main__":
    main()