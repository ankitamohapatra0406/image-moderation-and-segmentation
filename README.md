# image-moderation-and-segmentation
# 🧠 Machine Learning
This project uses a fine-tuned YOLOv8 model to detect unsafe objects (alcohol bottles,knives, etc.) in images.

⚙️ Model: YOLOv8 → detector_best.pt

🎓 Training: Done in ml/notebooks/train_detector.ipynb

🗂️ Dataset: YOLO-formatted images (ml/data/), detailed in dataset_info.md

🚀 Inference: Model detects unsafe regions, and the API blurs/flags them

🔌 Integration: Uses detector.py, segmenter.py, and pipeline.py
