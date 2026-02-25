import os
import yaml
from ultralytics import YOLO
import shutil
from sklearn.model_selection import train_test_split

def prepare_yolo_dataset(data_dir='data', output_dir='yolo_dataset'):
    """
    Converts collected images into YOLO format.
    Assumes data_dir contains subfolders named after classes.
    """
    if not os.path.exists(data_dir):
        print("No data found to train on.")
        return False
        
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not classes:
        print("No classes found.")
        return False

    # Create YOLO structure
    for split in ['train', 'val']:
        os.makedirs(f'{output_dir}/{split}/images', exist_ok=True)
        os.makedirs(f'{output_dir}/{split}/labels', exist_ok=True)

    class_map = {cls: i for i, cls in enumerate(classes)}
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if not images: continue
        
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
        
        for split, split_imgs in [('train', train_imgs), ('val', val_imgs)]:
            for img_name in split_imgs:
                src = os.path.join(cls_path, img_name)
                dst_img = f'{output_dir}/{split}/images/{cls}_{img_name}'
                shutil.copy(src, dst_img)
                
                # Create a "dummy" label (entire image as bounding box)
                # Since we don't have bounding boxes from just raw images, 
                # we assume the captured sample is the object itself.
                dst_label = f'{output_dir}/{split}/labels/{cls}_{img_name.rsplit(".", 1)[0]}.txt'
                with open(dst_label, 'w') as f:
                    # class_id center_x center_y width height (normalized)
                    f.write(f"{class_map[cls]} 0.5 0.5 1.0 1.0\n")

    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: cls for i, cls in enumerate(classes)}
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
        
    return True

def train_model():
    print("Preparing dataset...")
    if not prepare_yolo_dataset():
        return
        
    print("Starting fast-track training...")
    # Switching back to YOLOv8n (Nano) for much faster training
    model = YOLO('yolov8n.pt')
    
    # Optimized training for CPU speed and high accuracy
    model.train(
        data='data.yaml', 
        epochs=30,             # Increased slightly for better accuracy with faster learning rate
        imgsz=416,             # Reduced from 640 to 416 for significant CPU speedup
        batch=8,               # Optimized for CPU memory
        lr0=0.01,              # Increased initial learning rate for faster convergence
        project='fruit_runs', 
        name='custom_fruit',
        optimizer='AdamW',     # AdamW often converges faster on smaller datasets
        degrees=15,            # Rotation augmentation
        fliplr=0.5,            # Horizontal flip
        mosaic=1.0,            # Mosaic augmentation for better small object detection
        patience=8             # Balanced early stopping
    )
    
    # Move the best model to root
    best_model = 'fruit_runs/custom_fruit/weights/best.pt'
    if os.path.exists(best_model):
        shutil.copy(best_model, 'best.pt')
        print("Training complete. High-accuracy model saved as 'best.pt'")
    else:
        print("Training failed or stopped early.")

if __name__ == "__main__":
    train_model()
