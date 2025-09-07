# Deep Learning–Based Real-Time Detection of Monkey & Bison Intrusion and Humane Repellent System
This project presents a real-time animal detection system designed to help farmers protect their crops from monkey and bison intrusions.
The system uses YOLOv8 with a Swin Transformer backbone to continuously analyze video feeds and detect animals with high accuracy, even in complex farm environments.
When an intrusion is detected, the system is designed to trigger humane repellent mechanisms (such as ultrasonic deterrents, flashing lights, or alarms) while sending IoT-based alerts (SMS, notifications, or dashboard updates) to farmers for immediate action.
## Project Overview
Wildlife intrusion into farms causes significant crop loss. This project delivers a real-time detection and response pipeline: capture video → detect → decide → alert/repel → log events.
It is designed for PC/edge deployment with an optional cloud dashboard.

Target animals: Monkey, Bison
## Features
Real-time Object Detection with YOLOv8 + Swin Transformer

Improved accuracy: 90.2% mAP

Low-latency pipeline with OpenCV

Instant alerts via SMS/call/app (Twilio or custom)

Humane repellent: strobe lights, ultrasonic/loud sound

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
    E -->|Repel| G[Humane Repellent];
    D --> H[Logging + Cloud Sync];
## Hardware Requirements
Video Input: Pre-recorded or streaming video (no direct camera feed)

Compute: GPU-enabled PC (NVIDIA GPU recommended)

Repellent (any subset):

    Strobe/LED (via GPIO/relay)
    Sound driver / buzzer / speaker
    Optional ultrasonic module
Connectivity: Internet (for alerts/cloud), optional offline mode
## Model and Metrics
Backbone: Swin Transformer integrated into YOLOv8

Baseline: YOLOv8m

Updated accuracy with Swin Transformer:

    | Metric | Value | |-------------|---------| | Precision | 90.2% | | Recall | 72.4% | | mAP | 90.2% |
Confidence threshold: 0.8
## Key Elements
Directory Structure Explanation

    monkey-bison-guard/
    ├─ datasets/
    ├─ models/
    │  └─ best.pt
    ├─ src/
    │  ├─ train.py
    │  ├─ eval.py
    │  ├─ infer.py
    │  ├─ realtime.py
    │  ├─ alerts/
    │  │  └─ twilio_client.py
    │  ├─ repellent/
    │  │  ├─ gpio_relay.py
    │  │  └─ sound_player.py
    │  └─ utils/
    │     ├─ viz.py
    │     └─ logging.py
    ├─ configs/
    │  └─ default.yaml
    ├─ requirements.txt
    └─ README.md
## File Naming Conventions
    Model weights: best.pt, last.pt
    Dataset YAML: data.yaml
    Configs: descriptive YAML filenames (e.g., default.yaml)
    Scripts: action-oriented (train.py, infer.py, eval.py)
## Abbreviations and Glossary
    YOLO: You Only Look Once (object detection model)
    Swin: Shifted Window Transformer
    mAP: Mean Average Precision
    GPIO: General Purpose Input/Output
## Training
    yolo detect train   model=yolov8m.pt   data=datasets/wildlife/data.yaml   imgsz=640 epochs=100 batch=16 device=0   project=runs/train name=yolov8m_swin   save_period=1
## Evaluation
    yolo detect val   model=runs/train/yolov8m_swin/weights/best.pt   data=datasets/wildlife/data.yaml   imgsz=640 device=0
## Inference
    python src/infer.py --weights models/best.pt --source assets/test_video.mp4 --conf 0.8
## Real-Time Deployment
    python src/realtime.py --config configs/default.yaml --video assets/live_feed.mp4
## lerts & Humane Repellent
    Sound alerts implemented with Pygame mixer
    Flashlight/LED via relay (GPIO)
    Cooldown logic prevents repeated alerts while animal remains in frame
## Troubleshooting
    No last.pt → use save_period=1
    Resume training → resume=True model=path/to/last.pt
    Low FPS → reduce imgsz, use a smaller model (e.g., yolov8n.pt)
    False alarms → increase conf, check dataset labeling
## Contributors
    H. Chaithali Kini
    Hithashree B
    Pooja Nayak
    Soumya
Guide: Mr. Sunil Kumar S
## Acknowledgements
    Ultralytics YOLOv8
    Swin Transformer (Microsoft Research)
    OpenCV
    Roboflow (annotation/export)
    VTU & Department of AI & ML
