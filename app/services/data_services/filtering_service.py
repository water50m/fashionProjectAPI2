import pandas as pd
from typing import List, Optional
from app.services.config_service import load_config
from app.services.data_services.data_service import DataManager
from app.services.color_service import ColorCheck


class CSVSearchService:
    def __init__(self):
        self.config = load_config()
        self.rw = DataManager()
        self.color_check = ColorCheck()

    def filter_by_filename(self, df: pd.DataFrame, filenames: List[str]) -> pd.DataFrame:
        try:
            group_file = df.groupby('filename')
            selected_list = []

            for filename in filenames:
                try:
                    selected_file = group_file.get_group(filename)
                    selected_list.append(selected_file)
                except KeyError:
                    # ถ้าไม่มี filename นี้ใน group ก็ข้าม
                    continue

            if selected_list:
                return pd.concat(selected_list, ignore_index=True)
            else:
                return pd.DataFrame(columns=df.columns)
        except Exception as e:
            print(f"[filter_by_filename] Error: {e}")
            return pd.DataFrame(columns=df.columns)

    def filter_single_class(self, df: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
        try:
            group_class = df.groupby('class')
            selected_list = []
            for class_ in classes:
                try:
                    selected_file = group_class.get_group(class_)
                    selected_list.append(selected_file)
                except KeyError:
                    # ถ้าไม่มี filename นี้ใน group ก็ข้าม
                    continue

            if selected_list:
                return pd.concat(selected_list, ignore_index=True)
            else:
                return pd.DataFrame(columns=df.columns)

        except Exception as e:
            print(f"[filter_single_class] Error: {e}")
            return pd.DataFrame(columns=df.columns)

    def filter_collab_class(self, df: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
        """
        กรองเฉพาะ timestamp ที่มีครบทุก class ที่ระบุ
        """
        try:
            group_track = df.groupby('track_id')
            selected_list = []
            for track_id,obj_detail in group_track:
                group_class = obj_detail.groupby('class')
                if classes.issubset(group_class.groups.keys()):
                    selected_list.append(track_id)
            if selected_list:
                return pd.concat(selected_list, ignore_index=True)
            else:
                return pd.DataFrame(columns=df.columns)
                    
        except Exception as e:
            print(f"[filter_collab_class] Error: {e}")
            return pd.DataFrame(columns=df.columns)

    def filter_by_date(self, df: pd.DataFrame, date: Optional[str]) -> pd.DataFrame:
        try:
            if "date" not in df.columns or date is None:
                return df
            return df[df["date"] == date]
        except Exception as e:
            print(f"[filter_by_date] Error: {e}")
            return pd.DataFrame(columns=df.columns)

    def hex_to_rgb(self, hex_color: str):
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except Exception as e:
            print(f"[hex_to_rgb] Error: {e}")
            return (0, 0, 0)

    def bgr_to_rgb(self, bgr_str: str):
        try:
            nums = bgr_str.strip('[]').split()
            bgr = [float(n) for n in nums]
            return (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        except Exception as e:
            print(f"[bgr_to_rgb] Error: {e}")
            return (0, 0, 0)


    def filter_data(self, filter) -> pd.DataFrame:
        print(filter)
        try:
            data_loaded = self.rw.load_json(self.config.get("JSON_RESULT_PREDICT_CLOTHING",""))
            df = pd.DataFrame(data_loaded)
            # หาจากชื่อ file
            filtered =False
            if filter.filename:
                df = self.filter_by_filename(df, filter.filename)
                filtered = True

            if filter.class_color:
                cls = []
                cls_clr = []
                for item in filter.class_color:
                    
                    cls.append(item["classId"])
                    filtered_list = self.filter_single_class(df, [item["classId"]])

                    if item['colors']:
                        filtered_list = self.color_check.color_compare(filtered_list, item['colors'], threshold=100)
                    cls_clr.append(filtered_list)

                df = pd.concat(cls_clr)

                if filter.class_collab:
                    df = self.filter_collab_class(df, cls)
                filtered = True

            if df.empty:
                print("[search_csv] ไม่พบข้อมูลที่ตรงกับเงื่อนไข")
            else:
                print("[search_csv] พบผลลัพธ์")

            if filtered:
                return df.to_dict(orient="records")
            else:
                return []

        except Exception as e:
            print(f"[search_csv] Error: {e}")
            return []
