# Fruits Detection & Ripeness Identification System

A comprehensive AI-powered system to detect fruits, identify their species, and determine their ripeness level.

## Features
- **Real-time Tracking**: Uses YOLOv8 for high-speed fruit detection and tracking with visual bounding boxes.
- **Data Collection**: Integrated UI to capture image samples for manual "training" of new fruits or specific conditions.
- **Hybrid Recognition**: Seamlessly switches between built-in YOLO detections and user-defined fruit data.
- **Ripeness Analysis**: Color-based heuristic analysis for common fruits (Apples, Bananas, etc.).

## Project Components
1. **`main_data_creation.py`**: The "Developer" tool. Use this to:
   - Collect raw data for new fruit types.
   - Enter specific ripeness conditions (Ripe, Underripe, Overripe).
   - Trigger the training process to fine-tune the AI model.
2. **`main.py`**: The "User" application. A focused recognition interface showing fruit names and their conditions.
3. **`train.py`**: Automation script that handles dataset preparation and YOLOv8 fine-tuning.
4. **`utils.py`**: The core library containing the `Detector` class and ripeness detection logic.
5. **`database.json`**: Stores metadata for manually added fruits.

## Installation
Ensure you have Python 3.8+ installed, then run:
```bash
pip install ultralytics opencv-python customtkinter pillow scikit-learn PyYAML
```

## How to Proceed
1. **Gather Data**: Run `main_data_creation.py`, enter a fruit name/condition, and hit **Capture**.
2. **Train**: Click **Train Model** once you have captured enough samples.
3. **Deploy**: Use `main.py` for real-time identification of both default and custom fruits.

---
*Created for the Fruits Nutrition Detection Project*
