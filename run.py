import argparse
from pathlib import Path
from app.services.data_services.data_service import DataManager


dManage = DataManager()

def predict():
    from app.services.predict_service2 import processing_videos
    # deepsort
    # predict + tracking + predict clothing 
    processing_videos()

def main():
    # สร้าง parser
    parser = argparse.ArgumentParser(description="ตัวอย่าง argparse")
     # เพิ่ม argument
    parser.add_argument("--start", default=True, type=str, help="")
    parser.add_argument("--mode", type=str, help="predict, check, tune, search")
    parser.add_argument("--file", default=None, type=str, help="ที่อยู่ file video ถ้า mode คือ predict use result.json path for mode tune")
    parser.add_argument("--class1",type=str, nargs='+', help="class for search input like >> --class1 class color" )
    parser.add_argument("--class2",type=str, nargs='+', help="class for search input like >> --class2 class color" )


    # parse argument จาก command line
    args = parser.parse_args()
    file = args.file
    if args.start:
        if args.mode == 'check':
            from app.test.result_predict_test import processing_check
            processing_check()
        elif args.mode == 'predict2':
            from app.services.predict_service2 import processing_videos
            processing_videos()
        elif args.mode == 'tune':
            from app.services.tune_service import process
            process(path=file)
        elif args.mode == 'search':
            from app.services.search_service import search_process
            from app.services.play_service import show_resut
            cond = []
            class1 = args.class1
            if class1 is not None:
                class_item = class1.pop(0)
                colors  = class1
                cond.append([class_item,colors])

            class2 = args.class2
            if class2 is not None:
                class_item = class1.pop(0)
                colors  = class1
                cond.append([class_item,colors])
            read_path = Path(r"E:\ALL_CODE\python\fashion-project\resources\result_prediction\clothing_detection\results_clothing_detection_20250917_2.json")   
            data = dManage.load_json(str(read_path))

            print('showing state')
            result = search_process(cond,data)
            if len(data) == result.shape[0]:
                print('ไม่พบผลลัพที่ค้นหา')
                return
            else: 
                show_resut(result)

        else:
            from app.services.predict_service2 import processing_videos
            processing_videos()

if __name__ == "__main__":
    main()