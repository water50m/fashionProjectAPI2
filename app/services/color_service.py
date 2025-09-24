import pandas as pd
import cv2
import numpy as np


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

    def get_color_percentage_with_threshold(self, image, threshold=200):
        """   แยก object จาก background ด้วย threshold
        image: BGR image
        threshold: ค่าความสว่างเพื่อแยก background (0-255)\n    """  # inserted
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, object_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            total_object_pixels = cv2.countNonZero(object_mask)
            height, width = image.shape[:2]
            total_pixels = height * width  # จำนวน pixel ทั้งหมด
            total_background_pixels = total_pixels - total_object_pixels  # pixel ที่เป็นพื้นหลัง
            color_ranges = {
                    'Red': [(0, 50, 40), (10, 255, 255)],
                    'Red2': [(171, 50, 50), (180, 255, 255)],
                    'Yellow': [(26, 50, 40), (35, 255, 255)],
                    'LightGreen': [(36, 50, 40), (60, 255, 255)],
                    'Green': [(61, 50, 40), (85, 255, 255)],
                    'Cyan': [(86, 50, 40), (100, 255, 255)],
                    'Blue': [(101, 50, 40), (135, 255, 255)],
                    'Violet': [(136, 50, 40), (160, 255, 255)],
                    'Pink': [(161, 30, 150), (170, 255, 255)],
                    'White': [(0, 0, 200), (180, 30, 255)],
                    'Black': [(0, 0, 0), (180, 255, 50)],
                    'Orange': [(11, 50, 151), (25, 255, 255)],   # ส้มสด สว่าง
                    'Brown':  [(10, 100, 50), (25, 255, 150)],   # ส้มหม่น → น้ำตาล  
                    'Navy':   [(101, 80, 20), (130, 255, 100)]   # น้ำเงินเข้ม (ค่า V ต่ำ)
                }
            percentages_object = {}
            percentages_background = {}
            for color, (lower, upper) in color_ranges.items():
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                object_count = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=object_mask))
                if total_object_pixels > 0:
                    percentages_object[color] = (object_count / total_object_pixels) * 100
                else:
                    percentages_object[color] = 0
                background_mask = cv2.bitwise_not(object_mask)
                background_count = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=background_mask))
                percentages_background[color] = 100 if total_background_pixels > 0 else 0
            if 'Red' in percentages_object and 'Red2' in percentages_object:
                percentages_object['Red'] = percentages_object.pop('Red2')
                percentages_background['Red'] = percentages_background.pop('Red2')
            return (percentages_object, percentages_background)
        except Exception as e:
            print(f'\033[91m[get_color_percentage_with_threshold]\033[0m is error : {e}')


cls = ColorCheck()
if __name__== '__main__':
    print('c')
    import cv2
    pic_path = r"C:\Users\User\Downloads\lone_sleeve_top_red.jpg"
    cap = cv2.imread(pic_path)
    result = cls.get_color_percentage_with_threshold(cap)
    print(result[0])