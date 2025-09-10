import cv2
import numpy as np

def get_color_percentage_with_threshold(image, threshold=200):
    """
    แยก object จาก background ด้วย threshold
    image: BGR image
    threshold: ค่าความสว่างเพื่อแยก background (0-255)
    """
    # แปลง BGR -> HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # สร้าง mask ของ object ด้วย threshold (grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, object_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    # THRESH_BINARY_INV: ให้ object เป็นขาว, background เป็นดำ

    total_object_pixels = cv2.countNonZero(object_mask)
    total_pixels = image.shape[0] * image.shape[1]
    total_background_pixels = total_pixels - total_object_pixels

    # นิยามช่วงสี (HSV)
    color_ranges = {
        "Red":      [(0, 70, 50), (10, 255, 255)],
        "Red2":     [(170, 70, 50), (180, 255, 255)],
        "Orange":   [(11, 70, 50), (25, 255, 255)],
        "Yellow":   [(26, 70, 50), (35, 255, 255)],
        "LightGreen": [(36, 70, 50), (60, 255, 255)],
        "Green":    [(61, 70, 50), (85, 255, 255)],
        "Cyan/Blue": [(86, 70, 50), (125, 255, 255)],
        "Indigo":   [(126, 70, 50), (140, 255, 255)],
        "Violet":   [(141, 70, 50), (160, 255, 255)],
        "Magenta":  [(161, 70, 50), (169, 255, 255)],
        "Brown":    [(10, 100, 20), (30, 255, 200)],
        "White":    [(0, 0, 200), (180, 50, 255)],
        "Black":    [(0, 0, 0), (180, 255, 50)],
    }

    percentages_object = {}
    percentages_background = {}

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # mask & object_mask -> เฉพาะสีใน object
        object_count = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=object_mask))
        percentages_object[color] = (object_count / total_object_pixels) * 100 if total_object_pixels > 0 else 0

        # mask & background_mask -> เฉพาะสีใน background
        background_mask = cv2.bitwise_not(object_mask)
        background_count = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=background_mask))
        percentages_background[color] = (background_count / total_background_pixels) * 100 if total_background_pixels > 0 else 0

    # รวม Red + Red2
    if "Red" in percentages_object and "Red2" in percentages_object:
        percentages_object["Red"] += percentages_object.pop("Red2")
        percentages_background["Red"] += percentages_background.pop("Red2")

    return percentages_object, percentages_background

img = cv2.imread(r"C:\Users\User\Downloads\lone_sleeve_top_red.jpg")
obj_pct, bg_pct = get_color_percentage_with_threshold(img, threshold=200)
print(obj_pct)
print("Object colors %:")
for c, p in obj_pct.items():
    print(f"{c}: {p:.2f}%")

print("\nBackground colors %:")
for c, p in bg_pct.items():
    print(f"{c}: {p:.2f}%")