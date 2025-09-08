# Deep Learning–Based Real-Time Detection of Monkey & Bison Intrusion and Humane Repellent System
This project presents a real-time animal detection system designed to help farmers protect their crops from monkey and bison intrusions.
The system uses YOLOv8 with a Swin Transformer backbone to continuously analyze video feeds and detect animals with high accuracy, even in complex farm environments.
When an intrusion is detected, the system is designed to trigger humane repellent mechanisms (such as ultrasonic deterrents, flashing lights, or alarms) while sending IoT-based alerts (SMS, notifications, or dashboard updates) to farmers for immediate action.

## Project Overview
 The system provides a pipeline:
Capture video → Detect → Show Results → Notify via GUI indicators
It is designed for PC deployment.
Target Animals: Monkey, Bison

## Features
    Real-time object detection using YOLOv8 + Swin Transformer
    Low-latency processing with OpenCV
    Flashing GUI indicators for detected animals
    Modular GUI to select webcam, single video

## Technologies used
    Deep Learning: PyTorch + Ultralytics YOLOv8
    Backbone Model: Swin Transformer
    Programming Language: Python (packaged as .exe)
    Libraries: PyTorch, Ultralytics, OpenCV, NumPy
    Development Tools: VS Code, Google Colab
    
## System Architecture
graph TD;

    A[Video Input] --> B[Preprocessing];
    B --> C[YOLOv8 + Swin Transformer];
    C --> D[Detection Results];
    D --> E{GUI Indicators};
    
## Requirements
    Operating System: Windows 10 or higher
    Input: Webcam (for live detection) or video files (.mp4, .avi, .mov)
    Dependencies: Packaged in .exe, no Python installation required

## Hardware Requirements
    Video Input: Pre-recorded or streaming video (no direct camera feed)
    Compute: GPU-enabled PC (NVIDIA GPU recommended)

## Model and Metrics
    Backbone: Swin Transformer integrated into YOLOv8
    Baseline: YOLOv8n
    Updated accuracy with Swin Transformer:

    | Metric | Value | |-------------|---------| | Precision | 90.2% | | Recall | 72.4% | | mAP | 90.2% |

    Confidence threshold: 0.45

## Key Elements
Directory Structure Explanation
## 📂Project Structure
    Final_Projrct/
    ├─ monkey_bison_detection.exe     
    ├─ runs/detect/yolov8_swin_train1/weights/best.pt
    ├─ yolov8n.pt                   
    ├─ README.md
    
## Abbreviations and Glossary
    Abbreviations
    YOLO: You Only Look Once
    Swin: Shifted Window Transformer
    mAP: Mean Average Precision
    
## Running the Application
    Open the folder containing monkey_bison_detection.exe.
    Double-click the .exe to launch the GUI.
    Input Options:
    Use Webcam: Start live detection
    Browse Video File: Select a single video from your computer

## Detection Indicators
    Monkey Detection: Red flash on GUI when detected
    Bison Detection: Red flash on GUI when detected

## Note About Initial Windows
    When you click “Use Webcam” or “Browse Video File/Folder”, you may notice extra OpenCV windows briefly opening.
    Do not close the main GUI window — this is the interface with buttons and indicators.
    Close only the extra windows; detection will continue to work as expected.

## Optional Sample Videos
    Download sample videos for testing from Google Drive:
    Google Drive – Sample Videos
    Instructions:
    Click the link and download videos.
    Use Browse Video File or Browse Video Folder in the app to select them.

## Acknowledgements
We would like to express our heartfelt gratitude to our project guide and Head of the Department, Dr. Sunil Kumar S, Head of the Department, Artificial Intelligence & Machine Learning, for his valuable guidance, encouragement, and support throughout the course of this project. His constructive suggestions, positive attitude, and continuous motivation greatly helped us in coordinating and successfully completing this study, especially in preparing this report.

We would also like to acknowledge with deep appreciation the encouragement and support of our parents and friends, whose guidance and motivation were instrumental in completing this project.

Project Members:

H Chaithali Kini – 4MT22CI015

Hithashree B – 4MT22CI019

Pooja Nayak – 4MT22CI038

Soumya – 4MT22CI052
