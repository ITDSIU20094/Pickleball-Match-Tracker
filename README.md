# Pickleball Match Tracker

![Python](https://img.shields.io/badge/python-3.13-blue)
![Player Detection](https://img.shields.io/badge/player-YOLO11-green)
![Ball Detection](https://img.shields.io/badge/ball-YOLO11-yellow)
![Court Keypoints](https://img.shields.io/badge/court-ResNet50-red)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

> An end-to-end computer vision system for automated pickleball match analysis using player tracking, ball tracking, and court detection.

## ✨ Features

* 🎾 **Pickleball Ball Detection & Tracking** — Detect and track fast-moving pickleball across frames
* 🧍 **Player Detection & Tracking** — Maintain player identities during the match
* 🏟️ **Court Keypoint Detection** — Detect and reconstruct court geometry
* 📍 **Mini Court Visualization** — Transform gameplay into a top-down court view


## 🚀 Quick Start

### Prerequisites

* Python 3.13+
* Git
* CUDA (optional for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/ITDSIU20094/Pickleball-Match-Tracker.git

cd Pickleball-Match-Tracker

# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏗️ Project Structure
```
Pickleball-Match-Tracker/
    mini_court/                                 # Mini court visualization
        mini_court.py                           #   2D court transformation & ball/player overlay
    
    models/                                     # Pre-trained model weights
        best.pt                                 #   YOLOv8 object detection model
        keypoint14_model.pth                    #   ResNet-50 court keypoint detector (14 points)

    output/                                     # Generated outputs
        output_video.mp4                        #   Annotated match video with tracking overlays
    
    tracker/                                    # Object tracking & detection
        ball_tracker.py                         #   Ball detection & trajectory tracking
        player_tracker.py                       #   Player detection & identity persistence

    tracker_stubs/                              # Cached detections
        ball_detections.pkl                     #   Serialized ball detections across frames
        player_detections.pkl                   #   Serialized player detections across frames

    utils/                                      # Utility functions
        bbox_utils.py                           #   Bounding box operations & keypoint helpers
        conversions.py                          #   Pixel-to-meter coordinate conversion
        video_utils.py                          #   Video I/O & frame processing
    
    court_line_detector/                        # Court geometry detection
        court_line_detector.py                  #   ResNet keypoint detection model inference
    
    constants/                                  # Configuration constants
        __init__.py                             #   Project constants & parameters
    
    analysis/                                   # Data analysis notebooks
        ball_analysis.ipynb                     #   Post-match ball trajectory analysis
    
    preprocessing_dataset/                      # Dataset preparation
        convert_dataset.py                      #   Convert Roboflow dataset to COCO format
    
    main.py                                     # Main execution pipeline
    requirements.txt                            # Python dependencies
    
    # Training & Experimentation
    train-12keypoint-court.ipynb                # Train 12-point court keypoint model
    train-14keypoint-court.ipynb                # Train 14-point court keypoint model
    train-yolov8-object-detection-on-custom-dataset.ipynb    # Train YOLOv8 detector
    download_kaggle_dataset.ipynb               # Dataset download utilities
```

## 🏋️ Training
* Player and Ball with YOLO: train-yolov8-object-detection-on-custom-dataset.ipynb
* Pickleball court keypoint model: train-keypoint-court.ipynb

## 📄 Future work
* Player Pose Estimation & Fall Detection

Integrate YOLO Pose to estimate player body keypoints and analyze movement patterns for advanced action understanding. Future development will focus on detecting abnormal player motions such as sliding, falling, and collapse events during pickleball matches.

* Improve Ball Detection using TrackNet

Replace or combine the current ball detection pipeline with TrackNet to achieve more accurate and robust pickleball localization, especially under conditions of small object size, fast movement, motion blur, and temporary ball disappearance.

* Court Keypoint Refinement using Classical Computer Vision
    * Improve court keypoint precision through a post-processing pipeline:
    * Extract white court pixels from cropped court regions
    * Detect court boundary lines
    * Compute line intersections
    * Refine court keypoint coordinates based on geometric constraints

* Homography-based Keypoint Reconstruction
    * Enhance court robustness by reconstructing shifted or missing keypoints:
    * Estimate homography transformation from predicted points and reference points
    * Compare predicted geometry with the court template
    * Correct misplaced keypoints using homography projection
    * Improve stability under occlusion and partial court visibility

* Real-Time Match Analytics

Extend the system toward real-time inference and interactive performance analysis for live pickleball matches.

## 🎥 Demo Output
![Project Demo](docs/demo.gif)



