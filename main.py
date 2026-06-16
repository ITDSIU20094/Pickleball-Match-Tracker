import cv2
from utils import (read_video,
                   save_video)
from tracker import (PlayerTracker, BallTracker)
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

def main():
    input_video_path = r"C:\STUDY\Pre-Thesis\AI Robotics for Pickleball Match\Video\Video Project 8.mp4"
    video_frames = read_video(input_video_path)

    # Detect player 
    player_tracker = PlayerTracker(model_path="models/best.pt")
    player_detections = player_tracker.detect_frames(video_frames,
                                                    #  Run lần đầu thì để read_from_stub=False sau khi đã có file player_detections.pkl thì để True
                                                     read_from_stub=False, 
                                                    stub_path="tracker_stubs/player_detections.pkl")
    
    # # Detect ball
    ball_tracker = BallTracker(model_path="models/best.pt")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=False, 
                                                stub_path="tracker_stubs/ball_detections.pkl")
    # ball_detections = ball_tracker.interpolate_ball_positions(ball_detections) # Interpolate missing ball 

    # Detect court lines
    court_model_path = "models/keypoints14_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])  # Predict on the first frame for simplicity
    
    # Draw mini court
    mini_court = MiniCourt(video_frames[0], original_keypoints=court_keypoints)
    print("drawing_key_points sample:", mini_court.drawing_key_points[:10]) 

    # Detect ball shot
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print("Ball shot frames:", ball_shot_frames)
    # Detect bounce
    bounce_frames = ball_tracker.get_ball_bounce_frames(ball_detections)
    print("Bounce frames:", bounce_frames)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections, bounce_positions = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                                            ball_detections,
                                                                                                                            court_keypoints,
                                                                                                                            bounce_frames)
    # Draw output
    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    ## Draw ball bounding boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    ## Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    ## Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    # output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))
    # output_video_frames = mini_court.draw_bounce_points(output_video_frames, bounce_positions)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save output video
    save_video(output_video_frames, "output/output_video.mp4")

if __name__ == "__main__":
    main()   


# 3:12:51
