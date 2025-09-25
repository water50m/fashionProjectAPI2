import pandas as pd
import traceback
import numpy as np
import json
from pathlib import Path
from app.services.data_services.data_service import DataManager
from app.services.config_service import load_config

dManage = DataManager()
CLASS_NAMES_B = ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress']


def calculate_weight(list_data, weight):
    try:
        result = 0
        for i,item in enumerate(list_data):
            item_list = np.array(item)
            if item_list.max() == item_list.min():
                continue
            normalized1 = ((item_list - item_list.min()) / (item_list.max() - item_list.min())) * weight[i]
            result+= normalized1
        return result
    except Exception as e:
        print(f"[calculate_weight] is error : {e}")
def selet_data(data):
    try:    
        group_class = data.groupby('class_id')
        y = []
        x = []
        sum_confidence = []
        mean_confidence = []
        for group_name, group_df in group_class:
            if group_name == 'undifined':
                continue
            size = group_df.size
            sum_conf = group_df['confidence'].sum()/10
            mean_conf = (float(sum_conf)/float(size/100))
            sum_confidence.append(sum_conf)
            mean_confidence.append(mean_conf)
            y.append(size)
            x.append(group_name)

        weights_cal = calculate_weight([sum_confidence,y,mean_confidence],[0.7,0.3,1.3])
        top_idx = weights_cal.argsort()[::-1][:2]  # เอา index 2 ตัวที่มากที่สุด

        class_top1, class_top2 = x[top_idx[0]], x[top_idx[1]]
        return class_top1, class_top2
    except Exception as e:
         print(f"[selcet_data] is error : {e}")

def find_clothing( data):
        try:
            data_frame = pd.DataFrame(data)
            gruop_data = data_frame.groupby('track_id')
            body_T_class = [0, 1, 2, 3, 4, 5]
            body_B_class = [6, 7, 8]
            all_body_class = [9, 10, 11, 12]
            return_data = pd.DataFrame()
            current_group = 0
            count_group_track = len(gruop_data.size())
            for id, df in gruop_data:
                current_group+=1
                print(f'group {current_group}/{count_group_track} ID: {id}')
                nclass = df.groupby('class_id')
                if nclass.ngroups < 3:
                    return_data = pd.concat([df, return_data], ignore_index=True)
                    continue
                
                try:
                    top_classes = selet_data(df)
                    top1, top2 = top_classes
                except:
                    print(f'ID {id} พบข้อมูลไม่ถึง 2 class_id')
                    return_data = pd.concat([df, return_data], ignore_index=True)
                    continue


                if (top1 in all_body_class and top2 in all_body_class or (top1 in body_B_class or top2 in body_B_class)) and (top1 in all_body_class or top2 in all_body_class):
                    save_top1 = top1
                    save_top2 = top2
                    if save_top1 not in all_body_class:
                        top1 = save_top2
                        top2 = save_top1
                    
                    mask = df['class_id'].isin(all_body_class)
                    df.loc[mask, 'class_id'] = top1
                    false_list = df[~mask]
                    track_group = false_list.groupby('track_id')
                    new_rows = []
                    for track_id, objs in track_group:

                        frame_group = objs.groupby('frame')
                        for timeframe, frames in frame_group:
                            row = frames.iloc[0]
                            x_clothing, y_clothing, w_clothing, h_clothing = ([], [], [], [])
                            if row['class_id'] in body_T_class:
                                x_clothing = row['x_clothing']
                                y_clothing = row['y_clothing']
                                w_clothing = row['w_clothing']
                                h_clothing = 2  * row['h_clothing']
                            else:  # inserted
                                if row['class_id'] in body_B_class:
                                    x_clothing = row['x_clothing']
                                    y_clothing = row['y_clothing']+row['h_clothing']
                                    w_clothing = row['w_clothing']
                                    h_clothing = 2* row['h_clothing']
                            new_data1 = {**row.to_dict(), 
                                        'timestamp': timeframe, 
                                        'class_id': top1, 
                                        'class_name': CLASS_NAMES_B[top1] if top1 < len(CLASS_NAMES_B) else str(top1), 
                                        'x_clothing': x_clothing, 
                                        'y_clothing': y_clothing, 
                                        'w_clothing': w_clothing, 
                                        'h_clothing': h_clothing}
                            new_rows.append(new_data1)
                    new_df = pd.DataFrame(new_rows)
                    return_data = pd.concat([df, new_df, return_data], ignore_index=True)

                
                elif top1 in body_T_class and top2 in body_B_class:
                    df.loc[df['class_id'].isin(body_T_class), 'class_id'] = top1
                    df.loc[df['class_id'].isin(body_B_class), 'class_id'] = top2
                    mask = df['class_id'].isin(all_body_class)
                    true_list = df[mask]
                    track_group = true_list.groupby('track_id')
                    new_rows = []
                    for track_id, objs in track_group:

                        frame_group = objs.groupby('frame')
                        for timeframe, frames in frame_group:
                            row = frames.iloc[0]
                            new_data1 = {
                                            **row.to_dict(),
                                            'timestamp':timeframe , 
                                            'class_id': top1, 
                                            'class_name':  CLASS_NAMES_B[top1] if top1 < len(CLASS_NAMES_B) else str(top1), 
                                            'h_clothing': row['h_clothing'] / 2,
                                        }
                            new_data2 = {
                                            **row.to_dict(),
                                            'timestamp': timeframe,
                                            'class_id': top2,
                                            'class_name': CLASS_NAMES_B[top2] if top2 < len(CLASS_NAMES_B) else str(top2),
                                            'y_clothing': int(row['y_clothing']) - int(row['h_clothing']) /2,
                                            'h_clothing': int(row['h_clothing']) /2
                                        }                       
                            new_rows.append(new_data1)
                            new_rows.append(new_data2)
                    new_df = pd.DataFrame(new_rows)
                    return_data = pd.concat([df, new_df, return_data], ignore_index=True)

                elif top2 in body_T_class and top1 in body_B_class:
                    df.loc[df['class_id'].isin(body_T_class), 'class_id'] = top2
                    df.loc[df['class_id'].isin(body_B_class), 'class_id'] = top1
                    mask = df['class_id'].isin(all_body_class)
                    true_list = df[mask]
                    track_group = true_list.groupby('track_id')
                    new_rows = []
                    for track_id, objs in track_group:

                        frame_group = objs.groupby('frame')
                        for timeframe, frames in frame_group:
                            row = frames.iloc[0]
                            new_data1 = {
                                            **row.to_dict(),
                                            'timestamp': timeframe,
                                            'class_id': top2,
                                            'class_name': CLASS_NAMES_B[top2] if top2 < len(CLASS_NAMES_B) else str(top2),
                                            'h_clothing': int(row['h_clothing']) /2
                                        }

                            new_data2 = {
                                            **row.to_dict(),
                                            'timestamp': timeframe,
                                            'class_id': top1,
                                            'class_name': CLASS_NAMES_B[top1] if top1 < len(CLASS_NAMES_B) else str(top1),
                                            'y_clothing': int(row['y_clothing']) - int(row['h_clothing']) / 2,
                                            'h_clothing': int(row['h_clothing']) / 2
                                        }
                            new_rows.append(new_data1)
                            new_rows.append(new_data2)
                    new_df = pd.DataFrame(new_rows)
                    return_data = pd.concat([df, new_df, return_data], ignore_index=True)


                elif (top1 in body_T_class or top2 in body_T_class) and (top1 in all_body_class or top2 in all_body_class):
                    mask = df['class_id'].isin(body_B_class)
                    false_list = df[~mask]
                    true_list = df[mask]
                    class_id_top = top1 if top1 in body_T_class else top2
                    class_id_dress = top2 if top1 in body_T_class else top1
                    false_list.loc[false_list['class_id'].isin(body_T_class), 'class_id'] = class_id_top
                    false_list.loc[false_list['class_id'].isin(all_body_class), 'class_id'] = class_id_dress
                    df = pd.concat([false_list, true_list], ignore_index=True)
                    track_group = true_list.groupby('track_id')
                    new_rows = []
                    for track_id, objs in track_group:
                        frame_group = objs.groupby('frame')
                        for timeframe, frames in frame_group:
                            row = frames.iloc[0]
                            new_data1 = {
                                **row.to_dict(),
                                'timestamp': timeframe,
                                'class_id': class_id_top,
                                'class_name': CLASS_NAMES_B[class_id_top] if class_id_top < len(CLASS_NAMES_B) else str(class_id_top),
                                'y_clothing': int(row['y_clothing']) - int(row['h_clothing'])  # ✅ รวม y + h
                            }

                            new_data2 = {
                                **row.to_dict(),
                                'timestamp': timeframe,
                                'class_id': class_id_dress,
                                'class_name': CLASS_NAMES_B[class_id_dress] if class_id_dress < len(CLASS_NAMES_B) else str(class_id_dress),
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


def process(path=None,Data=None):
    print('tunning data')
    try:
        if path:
            read_path = Path(path)
        else:
            read_path = Path(r"E:\ALL_CODE\python\fashion-project\resources\result_prediction\clothing_detection\results_clothing_detection_20250917_2.json")
        
        if Data:
             detections = Data
        else:
            with open(read_path, "r", encoding="utf-8") as f:
                detections = json.load(f) 
        tuned = find_clothing(detections)

        # save result tuned 
        config = load_config()

        dir = config.get('RESULTS_PREDICT_DIR')
        file_readed = read_path.name
        dir = "resources/result_prediction/"
        save_dir = Path(dir) / "tuned_result"
        save_dir.mkdir(parents=True, exist_ok=True)  # สร้างโฟลเดอร์ (ถ้าไม่มี)

        save_path = save_dir / ("tuned_" + file_readed)
        print(save_path)
        tuned.to_json(save_path, orient="records", force_ascii=False, indent=4)
        return True

    except Exception as e:
        print(f"[process] is error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    process()