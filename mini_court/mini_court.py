import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_pixel_distance_to_meters,
    convert_meters_to_pixel_distance,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    get_bottom_center_of_bbox
)
from court_line_detector import CourtLineDetector

class MiniCourt():
    def __init__(self,frame, original_keypoints=None):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court=20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        self.H = None
        self.last_hitter = None
        if original_keypoints is not None:
            self.compute_homography(original_keypoints)

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.COURT_WIDTH,
                                                self.court_drawing_width
                                            )

    
    def set_court_drawing_key_points(self):

        drawing_key_points = [0] * 24

        left = int(self.court_start_x)
        right = int(self.court_end_x)
        center_x = int((left + right) / 2)

        court_height_px = self.convert_meters_to_pixels(
            constants.COURT_LENGTH
        )

        top = int(self.court_start_y)
        bottom = int(top + court_height_px)

        # Baseline -> Kitchen
        baseline_to_kitchen_px = self.convert_meters_to_pixels(
            constants.BASELINE_TO_KITCHEN
        )

        upper_kitchen_y = top + baseline_to_kitchen_px
        lower_kitchen_y = bottom - baseline_to_kitchen_px

        points = [
            # Baseline trên
            (left, top),
            (center_x, top),
            (right, top),

            # Kitchen trên
            (left, upper_kitchen_y),
            (center_x, upper_kitchen_y),
            (right, upper_kitchen_y),

            # Kitchen dưới
            (left, lower_kitchen_y),
            (center_x, lower_kitchen_y),
            (right, lower_kitchen_y),

            # Baseline dưới
            (left, bottom),
            (center_x, bottom),
            (right, bottom),
        ]

        idx = 0
        for x, y in points:
            drawing_key_points[idx] = x
            drawing_key_points[idx + 1] = y
            idx += 2

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),     
            (9, 11),    
            (0, 9),     
            (2, 11),    
            (3, 5),
            (6, 8),
            (1, 4),
            (7, 10)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
    
    def draw_court(self, frame):

        # Draw keypoints
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame,(x, y), 5, (0, 0, 255), -1)

        # Draw court lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        # Point layout:
        # 0  1  2
        # 3  4  5
        # 6  7  8
        # 9 10 11
        upper_kitchen_y = int(self.drawing_key_points[7])   # point 3 y
        lower_kitchen_y = int(self.drawing_key_points[13])  # point 6 y
        net_y = int((upper_kitchen_y + lower_kitchen_y) / 2)

        net_start_point = (
            int(self.drawing_key_points[0]),  # left boundary
            net_y
        )

        net_end_point = (
            int(self.drawing_key_points[4]),  # right boundary
            net_y
        )

        cv2.line(
            frame,
            net_start_point,
            net_end_point,
            (255, 0, 0),
            2
        )
        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def compute_homography(self, original_keypoints):
        # Thay các số 0,2,11,9 bằng chỉ số bạn quan sát được
        tl_idx = 0  # top-left
        tr_idx = 1  # top-right
        br_idx = 2 # bottom-right
        bl_idx = 3  # bottom-left
        src_pts = np.array([
            [original_keypoints[tl_idx*2], original_keypoints[tl_idx*2+1]],
            [original_keypoints[tr_idx*2], original_keypoints[tr_idx*2+1]],
            [original_keypoints[br_idx*2], original_keypoints[br_idx*2+1]],
            [original_keypoints[bl_idx*2], original_keypoints[bl_idx*2+1]]
        ], dtype=np.float32)
        dst_pts = np.array([
            [self.drawing_key_points[0], self.drawing_key_points[1]],      # top-left
            [self.drawing_key_points[4], self.drawing_key_points[5]],      # top-right
            [self.drawing_key_points[22], self.drawing_key_points[23]],    # bottom-right
            [self.drawing_key_points[18], self.drawing_key_points[19]]     # bottom-left
        ], dtype=np.float32)
        self.H, _ = cv2.findHomography(src_pts, dst_pts)
        print("Homography matrix:", self.H)

        for i, pt in enumerate(src_pts):
            pt_h = np.array([pt[0], pt[1], 1]).reshape(3,1)
            dst_h = self.H @ pt_h
            dst = dst_h[:2] / dst_h[2]
            print(f"src {i}: {pt} -> predicted dst: {dst.flatten()}, actual dst: {dst_pts[i]}")
    
    def get_mini_court_coordinates(self, object_position, original_court_key_points):
        # Chuyển original_court_key_points về list các điểm (x,y) - lấy tối đa 12 điểm
        if self.H is not None:
            pt = np.array([[object_position]], dtype=np.float32)
            mini_pt = cv2.perspectiveTransform(pt, self.H)
            return (float(mini_pt[0][0][0]), float(mini_pt[0][0][1]))
        else:
            # Fallback cho list/tuple
            points = [(original_court_key_points[i], original_court_key_points[i+1]) for i in range(0, min(24, len(original_court_key_points)), 2)]
        
        # Tính khoảng cách đến từng keypoint
        distances = []
        for idx, (kp_x, kp_y) in enumerate(points):
            dist = np.hypot(object_position[0] - kp_x, object_position[1] - kp_y)
            distances.append((dist, idx))
        
        # Lấy 4 điểm gần nhất
        distances.sort(key=lambda x: x[0])
        total_weight = 0.0
        mini_x, mini_y = 0.0, 0.0
        for dist, idx in distances[:4]:
            weight = 1.0 / (dist + 1e-6)
            total_weight += weight
            mini_x += weight * self.drawing_key_points[idx * 2]
            mini_y += weight * self.drawing_key_points[idx * 2 + 1]
        
        mini_x /= total_weight
        mini_y /= total_weight
        return (mini_x, mini_y)

    def convert_bounding_boxes_to_mini_court_coordinates(self,player_boxes,ball_boxes,original_court_key_points,bounce_frames=None):
        output_player_boxes = []
        output_ball_boxes = []
        bounce_positions = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = (ball_boxes[frame_num].get(1)
                if ball_boxes[frame_num] else None
            )
            frame_player_positions = {}
            # Convert player positions
            for player_id, data in player_bbox.items():
                foot_position = get_foot_position(data["bbox"])
                mini_pos = self.get_mini_court_coordinates(foot_position, original_court_key_points)
                frame_player_positions[player_id] = mini_pos  
            output_player_boxes.append(frame_player_positions)
            # Update last hitter
            if (ball_box is not None and len(player_bbox) > 0):
                ball_center = get_center_of_bbox(ball_box)
                distances = {}
                for pid, pdata in player_bbox.items():
                    foot = get_foot_position(pdata["bbox"])
                    distances[pid] = np.sqrt((ball_center[0]-foot[0])**2 +(ball_center[1]-foot[1])**2)

                nearest_player = min(distances,key=distances.get)

                if distances[nearest_player] < 180:
                    self.last_hitter = nearest_player

            # Draw ball on mini court
            if (ball_box is not None and self.last_hitter is not None and self.last_hitter in frame_player_positions):
                player_pos = frame_player_positions[self.last_hitter]
                player_class = player_bbox[self.last_hitter]["class_name"]
                offset = 12
                if player_class == "bg_player":
                    mini_ball = (player_pos[0], player_pos[1] + offset)
                else:
                    mini_ball = (player_pos[0], player_pos[1] - offset)
                output_ball_boxes.append({1: mini_ball})
            else:
                output_ball_boxes.append({})

            # Save bounce position
            if (ball_box is not None and bounce_frames is not None and frame_num in bounce_frames):
                bounce_center = get_center_of_bbox(ball_box)
                bounce_pos = (self.get_mini_court_coordinates(bounce_center,original_court_key_points))
                bounce_positions.append({"frame": frame_num,"position": bounce_pos})
        return (output_player_boxes, output_ball_boxes,bounce_positions)

            
    def draw_points_on_mini_court(self,frames,postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            # print(f"Drawing frame {frame_num}, positions dict: {postions[frame_num]}")
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames
    
    def draw_bounce_points(self, frames, bounce_positions):
        for bounce in bounce_positions:
            frame_idx = bounce["frame"]
            if frame_idx >= len(frames):
                continue
            x, y = bounce["position"]
            cv2.circle(frames[frame_idx],(int(x), int(y)), 12, (0,0,255), -1)
            cv2.putText(frames[frame_idx], "B", (int(x)+8, int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        return frames
    

