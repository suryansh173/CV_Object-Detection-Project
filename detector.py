import cv2
import torch
import csv
import os
import datetime
from ultralytics import YOLO
from collections import Counter

#paths
LOG_DIR      = "logs"
SNAPSHOT_DIR = "snapshots"
MODEL_PATH   = "models/yolov8m.pt"

os.makedirs(LOG_DIR,      exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

#load model on GPU
model = YOLO(MODEL_PATH)
model.to("cuda")

#open camera
# 0 = laptop cam
# 1 = external USB camera
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Camera not found. Check CAMERA_INDEX.")
    exit()

#csv log setup
log_filename = os.path.join(LOG_DIR, f"detections_{datetime.date.today()}.csv")
log_file     = open(log_filename, "w", newline="")
writer       = csv.writer(log_file)
writer.writerow(["timestamp", "class", "confidence", "x1", "y1", "x2", "y2"])

#fps calculation variables
fps         = 0
frame_count = 0
start_time  = datetime.datetime.now()

#confidence threshold
# only detections above this value will be shown
CONFIDENCE = 0.45

print("Detection running.")
print("Press S to save snapshot.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read frame from camera.")
        break

    #run inference on RTX 3050 GPU
    results = model(frame, conf=CONFIDENCE, verbose=False, device="cuda")

    #extract class names from detection results
    names        = results[0].names
    detected     = [names[int(c)] for c in results[0].boxes.cls]
    object_count = Counter(detected)

    #log every detection to csv file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        writer.writerow([
            timestamp,
            names[int(box.cls)],
            round(float(box.conf), 3),
            x1, y1, x2, y2
        ])
    log_file.flush()

    #calculate realtime fps 
    frame_count += 1
    elapsed      = (datetime.datetime.now() - start_time).total_seconds()
    if elapsed > 0:
        fps = round(frame_count / elapsed, 1)

    #draw bounding boxes on frame using ultralytics 
    annotated_frame = results[0].plot()

    #overlay fps counter top left
    cv2.putText(
        annotated_frame,
        f"FPS: {fps}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    #overlay detected object counts below fps
    y_offset = 80
    for obj, count in object_count.items():
        cv2.putText(
            annotated_frame,
            f"{obj}: {count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        y_offset += 30

    #display the annotated frame in a window
    cv2.imshow("CV Detection Project  |  S = Snapshot  |  Q = Quit", annotated_frame)

    #handle keyboard input
    key = cv2.waitKey(1) & 0xFF

    #S key saves current frame as jpg in snapshots folder
    if key == ord("s"):
        snapshot_path = os.path.join(
            SNAPSHOT_DIR,
            f"snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(snapshot_path, annotated_frame)
        print(f"Snapshot saved: {snapshot_path}")

    # Q key exits the loop cleanly
    if key == ord("q"):
        print("Quitting detection.")
        break

#release resources cleanly
cap.release()
log_file.close()
cv2.destroyAllWindows()
print(f"Log saved to: {log_filename}")