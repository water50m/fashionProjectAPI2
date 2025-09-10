import cv2
import numpy as np

color_ranges = {
    "Red":      [(0, 50, 40), (10, 255, 255)],
    "Orange":   [(11, 50, 128), (25, 255, 255)], 
    "Yellow":   [(26, 50, 40), (35, 255, 255)],
    "LightGreen": [(36, 50, 40), (60, 255, 255)],
    "Green":    [(61, 50, 40), (85, 255, 255)],
    "Cyan":     [(86, 50, 40), (100, 255, 255)], 
    "Blue":     [(101, 50, 40), (135, 255, 255)],
    "Violet":   [(136, 50, 40), (160, 255, 255)],
    "Pink":     [(161, 30, 150), (169, 255, 255)],
    "Red2":     [(170, 50, 40), (180, 255, 255)],
    "Brown":    [(10, 100, 20), (30, 255, 127)],
    "White":    [(0, 0, 200), (180, 30, 255)],
    "Gray":     [(0, 0, 51), (180, 30, 199)],
    "Black":    [(0, 0, 0), (180, 255, 39)],
}

hex_codes = {}

for name, (lower, upper) in color_ranges.items():
    Hmid = (lower[0] + upper[0]) // 2
    Smid = (lower[1] + upper[1]) // 2
    Vmid = (lower[2] + upper[2]) // 2

    hsv = np.uint8([[[Hmid, Smid, Vmid]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    hex_codes[name] = '#%02x%02x%02x' % tuple(rgb)

print(hex_codes)
