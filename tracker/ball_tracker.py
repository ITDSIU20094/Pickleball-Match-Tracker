from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        minimum_change_frames_for_hit = 15 # frames
        df_ball_positions['ball_hit'] = 0

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = (
                df_ball_positions['delta_y'].iloc[i] > 0 and
                df_ball_positions['delta_y'].iloc[i + 1] < 0
            )

            positive_position_change = (
                df_ball_positions['delta_y'].iloc[i] < 0 and
                df_ball_positions['delta_y'].iloc[i + 1] > 0
            )

            if negative_position_change or positive_position_change:
                change_count = 0

                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = (
                        df_ball_positions['delta_y'].iloc[i] > 0 and
                        df_ball_positions['delta_y'].iloc[change_frame] < 0
                    )

                    positive_position_change_following_frame = (
                        df_ball_positions['delta_y'].iloc[i] < 0 and
                        df_ball_positions['delta_y'].iloc[change_frame] > 0
                    )

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return frame_nums_with_ball_hits
    
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        class_names = results.names  # dict {0: 'ball', 1: 'bg_player', ...}

        ball_dict = {}
        for box in results.boxes:
            cls_id = int(box.cls.tolist()[0])
            if class_names[cls_id] == "ball":
                bbox = box.xyxy.tolist()[0]
                ball_dict[1] = bbox   # Gán ball với key = 1 (chỉ 1 ball mỗi frame)
                # Nếu có nhiều ball trong 1 frame, key sẽ bị ghi đè -> chỉ giữ ball cuối cùng
                # Bạn có thể đổi thành list nếu cần nhiều ball
        return ball_dict  

    def draw_bboxes(self,video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
    
    # def get_ball_bounce_frames(self, ball_positions):
    #     ball_positions = [x.get(1, []) for x in ball_positions]
    #     df = pd.DataFrame(
    #         ball_positions,
    #         columns=['x1','y1','x2','y2']
    #     )
    #     df['center_y'] = (
    #         df['y1'] + df['y2']
    #     ) / 2
    #     df['dy'] = df['center_y'].diff()
    #     bounce_frames = []
    #     for i in range(2, len(df)-2):
    #         if (
    #             df['dy'].iloc[i-1] > 0 and
    #             df['dy'].iloc[i] < 0
    #         ):
    #             bounce_frames.append(i)

    #     return bounce_frames

    def get_ball_bounce_frames(self, ball_positions):
        ball_positions = [
            x.get(1,[])
            for x in ball_positions
        ]

        df = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        df['cy']=(df['y1'] + df['y2']) /2
        df['cy']=(df['cy'].rolling(5).mean().bfill())
        df['vy']=(df['cy'].diff())
        bounce=[]
        for i in range(5, len(df)-5):
            before=(df['vy'].iloc[i-3:i].mean())
            after=(df['vy'].iloc[i+1:i+4].mean())
            if (before > 2 and after < -2):
                bounce.append(i)
        return bounce