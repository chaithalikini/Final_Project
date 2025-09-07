# Deep Learning–Based Real-Time Detection of Monkey & Bison Intrusion and Humane Repellent System
This project presents a real-time animal detection system designed to help farmers protect their crops from monkey and bison intrusions.
The system uses YOLOv8 with a Swin Transformer backbone to continuously analyze video feeds and detect animals with high accuracy, even in complex farm environments.
When an intrusion is detected, the system is designed to trigger humane repellent mechanisms (such as ultrasonic deterrents, flashing lights, or alarms) while sending IoT-based alerts (SMS, notifications, or dashboard updates) to farmers for immediate action.
## Project Overview
Wildlife intrusion into farms causes significant crop loss. This project delivers a real-time detection and response pipeline: capture video → detect → decide → alert → log events.
It is designed for PC/edge deployment .

Target animals: Monkey, Bison
## Features
Real-time Object Detection with YOLOv8 + Swin Transformer

Improved accuracy: 90.2% mAP

Low-latency pipeline with OpenCV

Instant alerts via beep

Modular codebase: train, evaluate, infer, deploy

Config-driven (YAML): thresholds, video paths, GPIO pins, alert settings
## Technologies used
Deep Learning : PyTorch + Ultralytics YOLOv8

Backbone Model : Swin Transformer (for high-resolution image analysis)

Programming Language : Python

Libraries/Frameworks : PyTorch, Ultralytics, OpenCV, NumPy, Pandas, Scikit-learn

Visualization : Matplotlib, Seaborn

Development Tools :VS Code, google colab.
## System Architecture
graph TD;

    A[Video Input] --> B[Preprocessing];
    B --> C[YOLOv8 + Swin Transformer];
    C --> D[Detection Results];
    D --> E{Decision};
    E -->|Alert| F[Notification System];
Requirements

Windows 10 or higher

Webcam (for live detection)

Optional: video files (.mp4, .avi, .mov)

All Python dependencies are packaged in the EXE, so no Python installation is required.

## Hardware Requirements
Video Input: Pre-recorded or streaming video (no direct camera feed)

Compute: GPU-enabled PC (NVIDIA GPU recommended)

Repellent (any subset):

    Sound driver / buzzer / speaker
Connectivity: Internet (for alerts/cloud), optional offline mode
## Model and Metrics
Backbone: Swin Transformer integrated into YOLOv8

Baseline: YOLOv8n

Updated accuracy with Swin Transformer:

    | Metric | Value | |-------------|---------| | Precision | 90.2% | | Recall | 72.4% | | mAP | 90.2% |

Confidence threshold: 0.8
## Key Elements
Directory Structure Explanation

   ## 📂 Project Structure

```bash
Final_Projrct/
├─ monkey_bison_detection.exe      # EXE file
├─ runs/detect/yolov8_swin_train1/weights/best.pt
├─ yolov8n.pt
├─ test_images/                    
├─ README.md


## Abbreviations and Glossary
    YOLO: You Only Look Once (object detection model)
    Swin: Shifted Window Transformer
    mAP: Mean Average Precision
    GPIO: General Purpose Input/Output
## alerts & Humane Repellent
    Sound alerts implemented with Pygame mixer
    Flashlight/LED via relay (GPIO)
    Cooldown logic prevents repeated alerts while animal remains in frame
## Results
Model Performance

    Metric	        Value
    mAP@0.5	        92.3%
    mAP@0.5:0.95	78.5%
    Precision	    90.1%
    Recall	        87.6%

Backbone: Swin Transformer integrated into YOLOv8

Baseline Model: YOLOv8m

Confidence Threshold: 0.8
## Troubleshooting
    No last.pt → use save_period=1
    Resume training → resume=True model=path/to/last.pt
    Low FPS → reduce imgsz, use a smaller model (e.g., yolov8n.pt)
    False alarms → increase conf, check dataset labeling
## Acknowledgements
We would like to express our heartfelt gratitude to our project guide and Head of the Department, Dr. Sunil Kumar S, Head of the Department, Artificial Intelligence & Machine Learning, for his invaluable guidance, encouragement, and support throughout the course of this project. His constructive suggestions, positive attitude, and continuous motivation greatly helped us in coordinating and successfully completing this study, especially in preparing this report.

We would also like to acknowledge with deep appreciation the encouragement and support of our parents and friends, whose guidance and motivation were instrumental in completing this project.

Project Members:

H Chaithali Kini – 4MT22CI015

Hithashree B – 4MT22CI019

Pooja Nayak – 4MT22CI038

Soumya – 4MT22CI052
