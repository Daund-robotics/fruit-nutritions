import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import time

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        # Prefer custom trained model if it exists
        if os.path.exists('best.pt'):
            print("Loading custom model: best.pt")
            self.model = YOLO('best.pt')
        else:
            print(f"Loading fast-track base model: {model_path}")
            self.model = YOLO(model_path)
            
        self.db_path = 'database.json'
        self.nutrition_path = 'nutrition_data.json'
        self.load_db()
        self.load_nutrition()
        
    def load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                self.db = json.load(f)
        else:
            self.db = {}

    def load_nutrition(self):
        if os.path.exists(self.nutrition_path):
            with open(self.nutrition_path, 'r') as f:
                self.nutrition = json.load(f)
        else:
            self.nutrition = {}

    def save_db(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f, indent=4)

    def detect_and_track(self, frame):
        # Using ByteTrack for better persistence during flips/rotations
        # Agnostic NMS helps prevent overlapping boxes for the same object seen as different classes
        results = self.model.track(
            frame, 
            persist=True, 
            verbose=False, 
            conf=0.25, 
            iou=0.5, 
            tracker="bytetrack.yaml",
            agnostic_nms=True
        )
        detections = []
        
        if results[0].boxes:
            for box in results[0].boxes:
                # Box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Class and confidence
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                # Check if this object is a fruit (default yolo coco classes)
                fruits = ['apple', 'orange', 'banana', 'broccoli', 'carrot']
                
                is_fruit = name in fruits or name in self.db
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'name': name,
                    'conf': conf,
                    'is_fruit': is_fruit
                })
                
        return detections

    def get_ripeness_color(self, fruit_crop, fruit_type):
        if fruit_crop.size == 0:
            return "Unknown"
            
        hsv = cv2.cvtColor(fruit_crop, cv2.COLOR_BGR2HSV)
        
        # Simple color-based logic for common fruits
        if fruit_type == 'apple':
            # Red/Green analysis
            red_lower = np.array([0, 100, 100])
            red_upper = np.array([10, 255, 255])
            green_lower = np.array([35, 100, 100])
            green_upper = np.array([85, 255, 255])
            
            red_mask = cv2.inRange(hsv, red_lower, red_upper)
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            red_ratio = np.sum(red_mask > 0) / fruit_crop.size
            green_ratio = np.sum(green_mask > 0) / fruit_crop.size
            
            if red_ratio > 0.05: return "Ripe (Red)"
            if green_ratio > 0.05: return "Underripe (Green)"
            return "Perfectly Ripe"

        elif fruit_type == 'banana':
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            yellow_ratio = np.sum(yellow_mask > 0) / fruit_crop.size
            
            if yellow_ratio > 0.05: return "Perfectly Ripe"
            return "Underripe"
            
        elif fruit_type == 'orange':
            orange_lower = np.array([10, 100, 100])
            orange_upper = np.array([25, 255, 255])
            orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
            orange_ratio = np.sum(orange_mask > 0) / fruit_crop.size
            if orange_ratio > 0.1: return "Perfectly Ripe"
            return "Underripe"

        return "Unknown"

    def save_fruit_data(self, fruit_name, condition, samples):
        # In a real app, we might train a model, here we just record the manual entry
        self.db[fruit_name.lower()] = {
            'condition': condition,
            'samples_count': len(samples),
            'last_updated': time.ctime()
        }
        self.save_db()
        
        # Save images to data folder
        os.makedirs(f'data/{fruit_name}', exist_ok=True)
        for i, img in enumerate(samples):
            # Save using a unique timestamp to avoid overwriting previous sessions
            ts = int(time.time() * 1000)
            cv2.imwrite(f'data/{fruit_name}/sample_{ts}_{i}.jpg', img)

    def reload_model(self):
        if os.path.exists('best.pt'):
            self.model = YOLO('best.pt')
            print("Model reloaded from best.pt")
        else:
            print("best.pt not found, keeping current model.")

    def get_nutrition(self, fruit_name):
        return self.nutrition.get(fruit_name.lower(), None)
