import cv2
import traceback
import pandas as pd
from app.services.config_service import load_config

def show_resut(data: pd.DataFrame):
    try:
        if data.empty:
            print('ไม่พบผลลัพที่คุณกำลังตามหา')
            return

        config = load_config()
        print('show data',data)
        video_name = pd.unique(data['filename'])
        video_file = video_name[0]

        group_ = data.groupby('track_id')
        cap = cv2.VideoCapture(config.get('"VIDEO_PATH"') + video_file)
        while True:
            for g_name,g_val in group_:
                for det in g_val:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, det['frame'])
                    ret, frame = cap.read()
                    x1, y1, w, h = det["x_person"], det["y_person"], det["w_person"], det["h_person"]
                    x2 = w + x1
                    y2 = h + y1
                    imcrop = frame[y2:y1, x2:x1]
                    if ret:
                        cv2.imshow(g_name, imcrop)

            # กด q เพื่อออก
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()       # ปล่อย resource
        cv2.destroyAllWindows()

    except Exception as e:
        print(f'[show_result] is error: {e}')
        traceback.print_exc()