import sys
import os
import cv2
from ultralytics import YOLO
from tkinter import Tk, Button, Label, Canvas, filedialog, messagebox
import threading
import winsound
import time

# ----------------------
# PyInstaller resource helper
# ----------------------
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# ----------------------
# YOLO Weights setup
# ----------------------
weights_path = resource_path(os.path.join("runs", "detect", "yolov8_swin_train1", "weights", "best.pt"))
model_animals = YOLO(weights_path)
model_coco = YOLO(resource_path("yolov8n.pt"))

# ----------------------
# Config
# ----------------------
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.3
NOTIFY_INTERVAL = 5      # seconds between notifications
INDICATOR_DURATION = 2   # seconds for GUI flash
FRAME_SKIP = 2           # process every 2nd frame
YOLO_INPUT_SIZE = 416    # resize for faster prediction

# ----------------------
# Notification tracker
# ----------------------
last_notification_time = {"Monkey": 0, "Bison": 0}
popup_active = {"Monkey": False, "Bison": False}

def notify_animal_thread(animal_name):
    def _notify():
        global popup_active
        if not popup_active[animal_name]:
            popup_active[animal_name] = True
            # Tkinter pop-up
            messagebox.showinfo(f"{animal_name} Detected", f"{animal_name} is in view!", parent=root)
            # Sound alert
            winsound.MessageBeep(winsound.MB_ICONASTERISK if animal_name=="Monkey" else winsound.MB_ICONEXCLAMATION)
            time.sleep(2)  # small delay before allowing next pop-up
            popup_active[animal_name] = False
    threading.Thread(target=_notify, daemon=True).start()

def notify_animal(animal_name):
    current_time = time.time()
    if current_time - last_notification_time[animal_name] >= NOTIFY_INTERVAL:
        notify_animal_thread(animal_name)
        last_notification_time[animal_name] = current_time
        flash_indicator(animal_name)

# ----------------------
# GUI indicator flash
# ----------------------
def flash_indicator(animal_name):
    def _flash():
        label = monkey_indicator if animal_name=="Monkey" else bison_indicator
        label.config(bg="red")
        time.sleep(INDICATOR_DURATION)
        label.config(bg="lightgrey")
    threading.Thread(target=_flash, daemon=True).start()

# ----------------------
# IoU filtering
# ----------------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter_area = max(0, x2-x1) * max(0, y2-y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area/union_area if union_area>0 else 0

def filter_boxes(animal_results, person_boxes):
    filtered = []
    for box in animal_results[0].boxes:
        animal_box = box.xyxy[0].cpu().numpy()
        keep = True
        for person_box in person_boxes:
            if iou(animal_box, person_box) > IOU_THRESHOLD:
                keep = False
                break
        if keep:
            filtered.append(box)
    return filtered

# ----------------------
# Global flags
# ----------------------
is_detecting = False
stop_detection_flag = False

# ----------------------
# Detection loop
# ----------------------
def run_detection(source=0):
    global is_detecting, stop_detection_flag
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open video source", parent=root)
        is_detecting = False
        return

    frame_count = 0
    while True:
        if stop_detection_flag:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        orig_height, orig_width = frame.shape[:2]
        frame_small = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))

        # Human detection
        coco_results = model_coco.predict(frame_small, conf=CONFIDENCE_THRESHOLD, verbose=False)
        person_boxes = [box.xyxy[0].cpu().numpy() for box in coco_results[0].boxes if int(box.cls[0])==0]

        # Animal detection
        animal_results = model_animals.predict(frame_small, conf=CONFIDENCE_THRESHOLD, verbose=False)
        custom_filtered = filter_boxes(animal_results, person_boxes)

        # Count animals
        monkey_count = sum(1 for box in custom_filtered if int(box.cls[0])==0)
        bison_count = sum(1 for box in custom_filtered if int(box.cls[0])==1)

        if monkey_count > 0: notify_animal("Monkey")
        if bison_count > 0: notify_animal("Bison")

        # Draw bounding boxes scaled back to original frame
        scale_x = orig_width / YOLO_INPUT_SIZE
        scale_y = orig_height / YOLO_INPUT_SIZE
        for box in custom_filtered:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
            label = "Monkey" if int(box.cls[0])==0 else "Bison"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Overlay counts
        cv2.putText(frame, f"Monkey: {monkey_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"Bison: {bison_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Monkey & Bison Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    is_detecting = False
    stop_detection_flag = False

# ----------------------
# GUI functions
# ----------------------
def start_detection(source):
    global is_detecting, stop_detection_flag
    if not is_detecting:
        is_detecting = True
        stop_detection_flag = False
        threading.Thread(target=run_detection, args=(source,), daemon=True).start()

def use_camera():
    start_detection(0)

def select_video():
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files","*.mp4 *.avi *.mov")]
    )
    if video_path:
        start_detection(video_path)

def stop_detection():
    global stop_detection_flag
    if is_detecting:
        stop_detection_flag = True

# ----------------------
# GUI setup
# ----------------------
root = Tk()
root.title("Monkey & Bison Detection")
root.geometry("350x280")
root.resizable(False, False)

Label(root,text="Select Input Source:",font=("Arial",12)).pack(pady=10)
Button(root,text="Use Webcam",command=use_camera,width=30,height=2).pack(pady=5)
Button(root,text="Select Video File",command=select_video,width=30,height=2).pack(pady=5)
Button(root,text="Stop Detection",command=stop_detection,width=30,height=2,bg="red",fg="white").pack(pady=10)

indicator_frame = Canvas(root,width=350,height=60)
indicator_frame.pack(pady=10)
Label(indicator_frame,text="Monkey Detection:",font=("Arial",10)).place(x=10,y=5)
monkey_indicator = Label(indicator_frame,bg="lightgrey",width=5,height=2)
monkey_indicator.place(x=150,y=5)
Label(indicator_frame,text="Bison Detection:",font=("Arial",10)).place(x=10,y=30)
bison_indicator = Label(indicator_frame,bg="lightgrey",width=5,height=2)
bison_indicator.place(x=150,y=30)

root.mainloop()
