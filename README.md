<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/efc3b36d-c440-4442-8544-f93975749c10" />🌦️ Smart Vision — Condition-Aware Object Detection
<p align="center">
  <img src="assets/smart-vision-overview.png" alt="Smart Vision project overview" width="100%">
</p>
<p align="center">
  <b>Adaptive Computer Vision for Real-World Weather & Low-Visibility Conditions</b>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/EfficientNet--B0-Condition%20Classifier-6f42c1">
  <img src="https://img.shields.io/badge/YOLO-Object%20Detection-111111">
  <img src="https://img.shields.io/badge/Real--ESRGAN-Super--Resolution-orange">
  <img src="https://img.shields.io/badge/SmolVLM-Visual%20Explanation-ffcc00">
</p>
---
🚀 What is Smart Vision?
Smart Vision is a condition-aware computer vision pipeline designed for images captured in challenging environments.
Instead of sending every image through the same processing path, the system first identifies the environmental condition and then decides whether enhancement is necessary.
It combines:
🧠 EfficientNet-B0 for environmental-condition classification
🎚️ Confidence Gate to avoid unnecessary processing
🛠️ Condition-specific preprocessing for difficult visual conditions
🔍 Real-ESRGAN for super-resolution
🎯 YOLO for object detection
💬 SmolVLM-500M-Instruct for natural-language visual explanation
Supported conditions
Condition	Purpose
🌫️ Fog	Handle reduced visibility
🌙 Low-Light	Improve dark-scene representation
☀️ Normal	Keep the original image
🌧️ Rain	Process rain-affected scenes
🏜️ Sand-Dust	Handle dusty environments
❄️ Snow	Process snow-affected scenes
---


---
✨ Why This Pipeline?
Traditional object detection pipelines often apply the same preprocessing to every image.
Smart Vision follows a different idea:
> **Understand the environment first, then adapt the vision pipeline.**
This makes the pipeline more suitable for:
🚗 Road-scene understanding
🌧️ Adverse-weather images
🌫️ Low-visibility environments
🚦 Intelligent transportation systems
🤖 Adaptive computer vision research
---
🔬 Core Workflow
1. Condition Classification
EfficientNet-B0 predicts one of six environmental conditions.
```text
Input Image
     ↓
Resize → 224 × 224
     ↓
ImageNet Normalization
     ↓
EfficientNet-B0
     ↓
Condition + Confidence
```
2. Confidence Gate
The system uses an 80% confidence threshold.
```text
Confidence ≥ 80%
        │
        ├── Normal → Original Image
        │
        └── Other condition → Enhancement Pipeline

Confidence < 80%
        │
        └── Fallback → Original Image
```
This prevents uncertain predictions from triggering unnecessary image processing.
3. Condition-Specific Enhancement
Different conditions receive different preprocessing.
```text
Low-Light  → Gamma / Contrast Enhancement
Fog        → Visibility Enhancement
Rain       → Rain-Streak Processing
Snow       → Contrast / Enhancement
Sand-Dust  → Contrast / Enhancement
Normal     → No Enhancement
```
4. Real-ESRGAN
After condition-specific processing, Real-ESRGAN can improve image resolution before detection.
```text
Processed Image
      ↓
Real-ESRGAN ×4
      ↓
Higher-Resolution Image
```
5. YOLO Detection
The enhanced image is passed to YOLO for object detection.
The project supports YOLO model weights such as:
YOLOv8s
YOLO11s
YOLO26s
The current configuration can be changed through `config.json`.
6. SmolVLM Explanation
SmolVLM receives the visual result and produces a concise explanation containing:
Environmental condition
Main detected objects
Effect of the condition on visibility/detection
The explanation is generated from the image and detection context rather than using a separate XAI/Grad-CAM module.
---
📊 Model Results
Environmental Condition Classifier
Model	Test Accuracy	Precision	Recall	F1
ResNet-18	81.75%	83.08	81.75	81.23
EfficientNet-B0	83.33%	83.31	83.33	83.10
EfficientNet-B0 was selected for the final condition-classification pipeline because it provided the stronger baseline accuracy with substantially fewer parameters.
YOLO Benchmark
A clean YOLO26s benchmark reported:
Metric	Score
Precision	86.49%
Recall	57.86%
mAP@50	70.21%
mAP@50–95	43.44%
> Results can change with dataset split, preprocessing, confidence threshold, model weights, and hardware.
---
🗂️ Project Structure
```text
smart-vision-condition-aware-yolo/
│
├── config.json
├── inference.py
├── requirements.txt
├── README.md
│
├── pipeline/
│   ├── classifier.py
│   ├── detector.py
│   ├── enhancers.py
│   ├── gate.py
│   ├── runner.py
│   └── vlm.py
│
├── models/
│   ├── classifier.pth
│   ├── yolo11s.pt
│   └── RealESRGAN_x4plus.pth
│
├── samples/
│   └── sample images
│
└── results/
    ├── processed_input.jpg
    ├── esrgan_output.jpg
    ├── detection_results.csv
    └── classifier_results.csv
```
> Large model weights are intentionally excluded from GitHub in most setups. Download or place them locally according to your project configuration.
---
⚙️ Configuration
Most pipeline settings are controlled from `config.json`.
Example:
```json
{
  "classes": [
    "Fog",
    "Low-Light",
    "Normal",
    "Rain",
    "Sand-Dust",
    "Snow"
  ],
  "confidence_threshold": 0.8,
  "classifier": {
    "arch": "efficientnet_b0",
    "input_size": 224
  },
  "detector": {
    "name": "YOLO11s",
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640
  },
  "vlm": {
    "enabled": true,
    "model": "HuggingFaceTB/SmolVLM-500M-Instruct"
  }
}
```
---
🛠️ Installation
1. Clone
```bash
git clone https://github.com/JANANI-RS06/smart-vision-condition-aware-yolo.git
cd smart-vision-condition-aware-yolo
```
2. Create virtual environment
Windows
```bash
python -m venv venv
venv\Scripts\activate
```
Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
---
▶️ Run Inference
Place a test image inside `samples/`.
Then run:
```bash
python inference.py "samples/image1.jpg"
```
For a different YOLO model:
```bash
python inference.py "samples/image1.jpg" yolo11s
```
The pipeline will display/save:
```text
Condition
Confidence
Gate Decision
Preprocessing Status
ESRGAN Status
Detected Objects
VLM Explanation
Inference Time
```
Output images and CSV files are stored in the configured `results/` directory.
---
💡 Example Output
```text
Condition      : Fog
Confidence     : 92.4%

Gate Decision  : PROCESSING APPLIED
Processing     : Fog enhancement

ESRGAN         : Applied

Detected       : car, truck, person

VLM Explanation:
The fog reduces visibility around the vehicles and person,
which can make object detection more challenging.
```
---
🧰 Technology Stack
Area	Technology
Language	Python
Deep Learning	PyTorch
Condition Classifier	EfficientNet-B0
Image Processing	OpenCV, NumPy, Pillow
Super-Resolution	Real-ESRGAN
Object Detection	Ultralytics YOLO
Vision-Language Model	SmolVLM-500M-Instruct
Data Handling	Pandas
Visualization	Matplotlib
---
🎯 Project Goals
Build a condition-aware computer vision pipeline
Improve detection under adverse visual conditions
Avoid unnecessary image enhancement
Compare detection performance across processing strategies
Generate human-readable visual explanations
Keep the pipeline modular and easy to experiment with
---
🔮 Future Improvements
[ ] Expand the adverse-weather dataset
[ ] Fine-tune the detector for weather-specific scenes
[ ] Optimize Real-ESRGAN inference speed
[ ] Add batch/video inference
[ ] Compare multiple YOLO variants systematically
[ ] Add automated evaluation reports
[ ] Deploy the final pipeline as an API
---
👩‍💻 Author
Lakshana Devi M
AI & Data Science Student  
Computer Vision • AI/ML • Generative AI
---
⭐ Acknowledgement
This project brings together open-source computer vision and vision-language technologies including PyTorch, Ultralytics YOLO, Real-ESRGAN, and SmolVLM.
If you find the project useful, consider giving the repository a ⭐.
---
📄 License
This project is intended for academic and research purposes.
See the repository license for usage details.
