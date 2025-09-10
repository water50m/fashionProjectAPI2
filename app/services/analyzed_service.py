import os 
import cv2
from collections import defaultdict
from services.data_services.data_service import DataManager 
import pandas as pd

manager = DataManager()

def extract_frame_and_crop(video_path: str, timestamp: float, x: int, y: int, w: int, h: int, 
                          output_path: str) -> bool:
    """Extract frame from video at timestamp and crop the specified region"""
    try:
        cap = cv2.VideoCapture(video_path)   # pylint: disable=no-member
        

        # Set video position to timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if ret:
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Ensure crop coordinates are within frame bounds
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            w = max(1, min(w, frame_width - x))
            h = max(1, min(h, frame_height - y))
            
            # Crop the image
            cropped = frame[y:y+h, x:x+w]
            
            # Save the cropped image
            cv2.imwrite(output_path, cropped)
            cap.release()
            return True
        
        cap.release()
        return False
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return False

def process_track_data(df: pd.DataFrame, video_name: str,video_path: str):
    """Process detection data for a specific video and generate track analysis"""
    
    # Filter data for the specific video
    video_df = df[df['filename'] == video_name].copy()

    if video_df.empty:
        return {
            "video_name": video_name,
            "tracks": [],
            "all_classes": [] 
        }
    
    tracks_data = []
    all_classes_set = set()
    
    # Group by track_id
    for track_id, track_group in video_df.groupby('track_id'):
        # Calculate total confidence for each class
        class_confidences = defaultdict(float)
        class_data = defaultdict(list)
        
        for _, row in track_group.iterrows():
            class_name = row['class_name']
            confidence = row['confidence']
            # print("row",row)
            class_confidences[class_name] += confidence
            try:
                class_data[class_name].append({
                'timestamp': row['timestamp'],
                'confidence': confidence,
                'x_person': row['x_person'],
                'y_person': row['y_person'],
                'w_person': row['w_person'],
                'h_person': row['h_person']
            })
            except Exception as e:
                print(f"Error processing track data: {e}")
            all_classes_set.add(class_name)

        # Get top 2 classes by total confidence
        sorted_classes = sorted(class_confidences.items(), key=lambda x: x[1], reverse=True)
 
        top_classes = [cls[0] for cls in sorted_classes[:2]]

        if not top_classes:
            continue
            
        # Find best timestamp for the top class
        best_class = top_classes[0]
        best_detection = max(class_data[best_class], key=lambda x: x['confidence'])
        best_timestamp = best_detection['timestamp']
        

        # Generate example image
        image_filename = f"{os.path.splitext(video_name)[0]}-{track_id}-{best_timestamp:.1f}-{best_class.replace(' ', '_')}.jpg"
        image_path = os.path.join(config.get("EXAMPLE_PICS_PATH", ""), image_filename)

        
        # Extract and crop frame
        success = extract_frame_and_crop(
            video_path,
            best_timestamp,
            int(best_detection['x_person']),
            int(best_detection['y_person']),
            int(best_detection['w_person']),
            int(best_detection['h_person']),
            image_path
        )
        
        # Calculate time range for this track
        timestamps = track_group['timestamp'].values
        time_range = {
            'start': float(timestamps.min()),
            'end': float(timestamps.max())
        }
        
        tracks_data.append({
            'track_id': int(track_id),
            'classes': top_classes,
            'total_confidences': {cls: float(conf) for cls, conf in class_confidences.items()},
            'best_timestamp': float(best_timestamp),
            'best_class': best_class,
            'time_range': time_range,
            'example_image_url': f"{config.get('FAST_API', 'http://localhost:8000')}/example_pics/{image_filename}" if success else None,
            'bbox': {
                'x': int(best_detection['x_person']),
                'y': int(best_detection['y_person']),
                'w': int(best_detection['w_person']),
                'h': int(best_detection['h_person'])
            }
        })
    print("for track_id, track_group in video_df.groupby('track_id'):")
    return {

        'video_name': video_name,
        'tracks': tracks_data,
        'all_classes': list(all_classes_set)
    }


def analyze_result_predict_data(data):
    try:
        df = pd.DataFrame(manager.load_json(data.predictionStrategyClothing))
        if df:
            result = process_track_data(df, data.file_name,data.file_path)
            try:
                cap = cv2.VideoCapture(data.file_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                result['video_duration'] = duration
                cap.release()
            except:
                pass
        
            return result
    except Exception as e:
        print(f"[analyze_result_predict_data] is error : {e}")