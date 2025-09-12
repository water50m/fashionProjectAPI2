from lib.boxmot.boxmot.trackers.bytetrack.bytetrack import ByteTrack
import torch


def bytetrack(data,frame):
    tracker = ByteTrack(track_thresh=0.5, match_thresh=0.8, frame_rate=30)
    xyxy = data.boxes.xyxy.cpu()   # [num_objects, 4]
    conf = data.boxes.conf.cpu()   # [num_objects]
    cls = data.boxes.cls.cpu() 
    detections = torch.cat([xyxy, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1).numpy()
    # input  list ของ [x1, y1, x2, y2, conf, class_id] , frame
    return tracker.update(detections, frame) 



def usethis(data,frame,typetrack):
    if typetrack==0:
        # output : x1, y1, x2, y2, track_id, conf, cls, det_ind
        return bytetrack(data,frame)

