from lib.boxmot.boxmot.trackers.bytetrack.bytetrack import ByteTrack
from lib.boxmot.boxmot.trackers.strongsort.strongsort import StrongSort
from lib.boxmot.boxmot.trackers.boosttrack.boosttrack import BoostTrack
from lib.boxmot.boxmot.trackers.botsort.botsort import BotSort
from lib.boxmot.boxmot.trackers.ocsort.ocsort import OcSort
import torch
from pathlib import Path

boosttrack_tracker = BoostTrack(
                                    reid_weights=Path('resources\models\osnet_x1_0_msmt17.pt'),  # ถ้าใช้ reid
                                    device="0",
                                    half=False,
                                    max_age=30,
                                    with_reid=False,
                                    per_class=False
                                )
bytetrack_tracker = ByteTrack(track_thresh=0.5, match_thresh=0.8, frame_rate=30)
strongsort_tracker = StrongSort(
                                    reid_weights=Path('resources\models\osnet_x1_0_msmt17.pt'),  # ReID model for StrongSORT
                                    device="0",  # or "cpu"
                                    half=False
                                )

botsort_tracker = BotSort( reid_weights=Path('resources\models\osnet_x1_0_msmt17.pt'),
                          device="0",
                          half=False,
                          with_reid=False
                          )

ocsort_tracker = OcSort()

# 3/5
def bytetrack(data,frame):
    
    xyxy = data.boxes.xyxy.cpu()   # [num_objects, 4]
    conf = data.boxes.conf.cpu()   # [num_objects]
    cls = data.boxes.cls.cpu() 
    detections = torch.cat([xyxy, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1).numpy()
    # input  list ของ [x1, y1, x2, y2, conf, class_id] , frame
    return bytetrack_tracker.update(detections, frame) 
# 0/5
def strongsort(data,frame):
    xyxy = data.boxes.xyxy.cpu()   # [num_objects, 4]
    conf = data.boxes.conf.cpu()   # [num_objects]
    cls = data.boxes.cls.cpu() 
    detections = torch.cat([xyxy, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1).numpy()
    return strongsort_tracker.update(detections, frame)
# 2/5
def boosttrack(data,frame):
    xyxy = data.boxes.xyxy.cpu()   # [num_objects, 4]
    conf = data.boxes.conf.cpu()   # [num_objects]
    cls = data.boxes.cls.cpu() 
    detections = torch.cat([xyxy, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1).numpy()
    return boosttrack_tracker.update(detections, frame)

def botsort(data,frame):
    xyxy = data.boxes.xyxy.cpu()   # [num_objects, 4]
    conf = data.boxes.conf.cpu()   # [num_objects]
    cls = data.boxes.cls.cpu() 
    detections = torch.cat([xyxy, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1).numpy()
    return botsort_tracker.update(detections, frame)

def ovsort(data,frame):
    xyxy = data.boxes.xyxy.cpu()   # [num_objects, 4]
    conf = data.boxes.conf.cpu()   # [num_objects]
    cls = data.boxes.cls.cpu() 
    detections = torch.cat([xyxy, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1).numpy()
    return ocsort_tracker.update(detections, frame)

def usethis(data,frame,typetrack):
    if typetrack==0 : 
        # output : x1, y1, x2, y2, track_id, conf, cls, det_ind
        return bytetrack(data,frame)
    elif typetrack==1:
        # output : x1, y1, x2, y2, track_id, conf, cls, det_ind
        return strongsort(data,frame)
    elif typetrack==2:
        # Returns:
        #   np.ndarray: Tracked objects in the format
        #               [x1, y1, x2, y2, id, confidence, cls, det_ind]
        #               (with cls and det_ind set to -1 if unused)
        return boosttrack(data,frame)
    
    elif typetrack ==3:
        
        return botsort(data,frame)
    
    elif typetrack ==4:
        return ovsort(data,frame)