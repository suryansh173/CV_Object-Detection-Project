# Real-Time Object Detection System

A real-time object detection and classification system built using YOLOv8 
and OpenCV, running on NVIDIA GPU with CUDA acceleration.

## Features

- Real-time object detection using YOLOv8m model
- GPU accelerated inference on NVIDIA RTX 3050
- Live bounding boxes with class labels and confidence scores
- FPS counter displayed on screen
- Object count per class displayed live
- Press S to save snapshot of current frame
- Press Q to quit cleanly
- Automatic CSV logging of all detections with timestamp

## Tech Stack

- YOLOv8 (Ultralytics)
- OpenCV 4.x
- PyTorch 2.x with CUDA 12.4
- Python 3.12

## Setup Instructions

### 1. Clone the repository

git clone https://github.com/yourusername/CV_Detection_Project.git
cd CV_Detection_Project

### 2. Create virtual environment

python -m venv venv
venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

Note: The above installs CPU version of PyTorch.
For GPU support with CUDA 12.4 run this instead:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

### 4. Download YOLOv8 model

python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
move yolov8m.pt models\yolov8m.pt

### 5. Run the detector

python detector.py

## Controls

- S key — save snapshot to snapshots folder
- Q key — quit the application

## Project Structure

CV_Detection_Project/
    assets/          — test images or videos
    logs/            — auto generated CSV detection logs
    models/          — YOLOv8 model weights (not pushed to GitHub)
    snapshots/       — auto saved snapshots on S key press
    detector.py      — main detection script
    requirements.txt — python dependencies
    README.md        — project documentation

## Future Scope

This project forms the base layer for a Construction Site PPE Safety Monitor
that detects helmet and safety vest violations in real time using the same
YOLOv8 pipeline fine-tuned on a construction PPE dataset.

## AI Assistance

This project was developed with the assistance of Claude (claude.ai) by Anthropic.
Claude was used for:

- Architecture planning and technology stack decisions
- Code structuring and commenting best practices
- Debugging CUDA and PyTorch installation issues


All code was implemented, tested, and validated on a local development environment with NVIDIA RTX 3050 GPU.

## Research Papers

The future scope of this project — Construction Site PPE Safety Monitor —
is grounded in the following peer-reviewed research:

1. Lee, J. & Lee, S. (2023). Construction Site Safety Management: A Computer
   Vision and Deep Learning Approach. Sensors, 23(2), 944.
   https://doi.org/10.3390/s23020944

2. Ferdous, M. & Ahsan, S.M.M. (2022). PPE detector: a YOLO-based architecture
   to detect personal protective equipment for construction sites.
   PeerJ Computer Science.
   https://doi.org/10.7717/peerj-cs.999

3. Elhadad, M.M. et al. (2024). Enhancing Construction Site Safety Using AI:
   Custom YOLOv8 Model for PPE Compliance Detection.
   Proceedings of the 2024 EC3 Conference.

4. Delhi, V.S.K., Sankarlal, R. & Thomas, A. (2020). Detection of PPE
   Compliance on Construction Site Using Computer Vision Based Deep Learning
   Techniques. Frontiers in Built Environment, 6, 136.

5. Singh, S. et al. (2024). PPE Detection for Construction Site Safety
   Leveraging YOLOv8. Proceedings of ICICC 2024. SSRN: 4923318.

6. Journal of Construction Engineering and Management (2025). Enhancing
   Worker Safety: Real-Time Automated Detection of PPE to Prevent Falls
   from Heights Using Improved YOLOv8 and Edge Devices.
   ASCE Library, Vol. 151, No. 1.
