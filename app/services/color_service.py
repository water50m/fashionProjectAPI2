import pandas as pd


class ColorCheck:
    def color_distance(self, rgb1, rgb2):
        try:
            return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5
        except Exception as e:
            print(f"[color_distance] Error: {e}")
            return float("inf")

    def prepare_and_find_similar_colors(self, df: pd.DataFrame, input_colors_hex: list[str], threshold: float):
        try:
            input_colors_rgb = [self.hex_to_rgb(c) for c in input_colors_hex]
            df_filtered = df.copy()
            df_filtered["mean_color_rgb"] = df_filtered["mean_color_bgr"].apply(self.bgr_to_rgb)

            result_rows = []
            for rgb in input_colors_rgb:
                for _, row in df_filtered.iterrows():
                    dist = self.color_distance(rgb, row['mean_color_rgb'])
                    if dist <= threshold:
                        result_rows.append(row)

            if result_rows:
                return pd.DataFrame(result_rows)
            else:
                return pd.DataFrame(columns=df.columns)
        except Exception as e:
            print(f"[prepare_and_find_similar_colors] Error: {e}")
            return pd.DataFrame(columns=df.columns)
        
    def color_compare(self, df: dict, color_selected: list[str]):
        """
        ค้นหาว่า object ที่ได้จากการ predict มีอันไหนที่มีค่าสี ตรงกับที่ถูกเลือกมาบ้าง(ต้องมี%พึ้นที่มากกว่าค่า threshold )
        
        """

        try:
            threshold = 60
            if len(color_selected) == 2:
                threshold = 40
            elif len(color_selected) == 3:
                threshold = 30

            selected_list = []
            for obj in df:   # df เป็น dict {filename: [objs]}
                    for color in color_selected:
                        if obj.get("colors").get(color," ") > threshold:
                            selected_list.append(obj)
                            
            return pd.DataFrame(selected_list)
        except Exception as e:
            print(f"[\033[91mprepare_and_find_similar_colors\033[0m] is error : {e}")