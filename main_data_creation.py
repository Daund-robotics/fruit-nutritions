import tkinter as tk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
from utils import Detector
import os
import subprocess
import sys

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Fruit Detection & Data Creation Tool")
        self.geometry("1100x700")

        self.detector = Detector()
        self.cap = cv2.VideoCapture(0)
        self.is_capturing = False
        self.samples = []
        self.current_frame = None
        
        # UI Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar (Navigation)
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Fruit AI", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20)
        
        self.add_data_btn = ctk.CTkButton(self.sidebar, text="Add Data", command=self.show_add_data)
        self.add_data_btn.pack(pady=10, padx=10)
        
        self.recognize_btn = ctk.CTkButton(self.sidebar, text="Recognize", command=self.show_recognize)
        self.recognize_btn.pack(pady=10, padx=10)

        # Main Area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=1)

        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Control Panel (Right side)
        self.control_panel = ctk.CTkFrame(self.main_frame)
        self.control_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.mode = "add_data" # Default mode
        self.setup_add_data_ui()
        
        self.update_video()

    def setup_add_data_ui(self):
        # Clear control panel
        for widget in self.control_panel.winfo_children():
            widget.destroy()
            
        ctk.CTkLabel(self.control_panel, text="Add Data Mode", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        ctk.CTkLabel(self.control_panel, text="Fruit Name:").pack(pady=(10, 0))
        self.fruit_name_entry = ctk.CTkEntry(self.control_panel)
        self.fruit_name_entry.pack(pady=5, padx=10, fill="x")
        
        ctk.CTkLabel(self.control_panel, text="Condition:").pack(pady=(10, 0))
        ctk.CTkLabel(self.control_panel, text="(Ripe/Underripe/Overripe)", font=ctk.CTkFont(size=10)).pack()
        self.condition_entry = ctk.CTkEntry(self.control_panel)
        self.condition_entry.pack(pady=5, padx=10, fill="x")
        
        self.sample_label = ctk.CTkLabel(self.control_panel, text="Samples taken: 0")
        self.sample_label.pack(pady=20)
        
        self.capture_btn = ctk.CTkButton(self.control_panel, text="Capture", fg_color="green", command=self.start_capture)
        self.capture_btn.pack(pady=5, padx=10, fill="x")
        
        self.stop_btn = ctk.CTkButton(self.control_panel, text="Stop", fg_color="red", command=self.stop_capture)
        self.stop_btn.pack(pady=5, padx=10, fill="x")
        
        self.save_btn = ctk.CTkButton(self.control_panel, text="Save Data", command=self.save_data)
        self.save_btn.pack(pady=10, padx=10, fill="x")
        
        ctk.CTkLabel(self.control_panel, text="--- OR ---").pack(pady=5)
        
        self.train_btn = ctk.CTkButton(self.control_panel, text="Train Model", fg_color="purple", command=self.start_training)
        self.train_btn.pack(pady=10, padx=10, fill="x")

    def setup_recognize_ui(self):
        for widget in self.control_panel.winfo_children():
            widget.destroy()
            
        ctk.CTkLabel(self.control_panel, text="Recognition Mode", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.res_name_label = ctk.CTkLabel(self.control_panel, text="Fruit: Waiting...", font=ctk.CTkFont(size=18))
        self.res_name_label.pack(pady=20)
        
        self.res_cond_label = ctk.CTkLabel(self.control_panel, text="Condition: -", font=ctk.CTkFont(size=18))
        self.res_cond_label.pack(pady=10)
        
        self.nutrition_panel = ctk.CTkTextbox(self.control_panel, height=200, width=200)
        self.nutrition_panel.pack(pady=10, padx=10)
        self.nutrition_panel.insert("0.0", "Nutritional Info:\nSelect a fruit...")

    def show_add_data(self):
        self.mode = "add_data"
        self.setup_add_data_ui()

    def show_recognize(self):
        self.mode = "recognize"
        self.setup_recognize_ui()

    def start_capture(self):
        self.is_capturing = True
        self.samples = []

    def stop_capture(self):
        self.is_capturing = False

    def save_data(self):
        name = self.fruit_name_entry.get()
        cond = self.condition_entry.get()
        if not name or not cond:
            print("Please enter name and condition")
            return
        
        if not self.samples:
            print("No samples captured")
            return
            
        self.detector.save_fruit_data(name, cond, self.samples)
        self.samples = []
        self.sample_label.configure(text="Samples taken: 0")
        print(f"Data saved for {name}. Starting training automatically...")
        
        # Automatically trigger training
        self.start_training()

    def start_training(self):
        self.train_btn.configure(state="disabled", text="Training...")
        threading.Thread(target=self.run_train_script, daemon=True).start()

    def run_train_script(self):
        try:
            # Run the train.py script
            result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True)
            print(result.stdout)
            if result.returncode == 0:
                print("Training finished successfully.")
                self.detector.reload_model()
                self.after(0, lambda: self.train_btn.configure(state="normal", text="Training Done!", fg_color="green"))
            else:
                print(f"Training failed: {result.stderr}")
                self.after(0, lambda: self.train_btn.configure(state="normal", text="Error in Training", fg_color="red"))
        except Exception as e:
            print(f"Error starting training: {e}")
            self.after(0, lambda: self.train_btn.configure(state="normal", text="Error", fg_color="red"))

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            detections = self.detector.detect_and_track(frame)
            
            display_frame = frame.copy()
            
            found_fruit = None
            
            for d in detections:
                if d['is_fruit']:
                    x1, y1, x2, y2 = d['bbox']
                    # Requirement: Blue square for tracking in add_data
                    # Use a generic label in add_data mode to avoid confusion with default classes
                    label = "Tracking Object..." if self.mode == "add_data" else f"{d['name']} {d['conf']:.2f}"
                    color = (255, 0, 0) if self.mode == "add_data" else (0, 255, 0)
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    found_fruit = d
                    
                    if self.is_capturing:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            self.samples.append(crop)
                            self.sample_label.configure(text=f"Samples taken: {len(self.samples)}")

            if self.mode == "recognize" and found_fruit:
                name = found_fruit['name']
                # Requirement: Only show the train data from the database
                if name.lower() in self.detector.db:
                    condition = self.detector.db[name.lower()]['condition']
                    display_name = name
                    
                    self.res_name_label.configure(text=f"Fruit: {display_name.capitalize()}")
                    self.res_cond_label.configure(text=f"Condition: {condition}")
                    
                    # Update nutrition info
                    nutri = self.detector.get_nutrition(display_name)
                    if nutri:
                        info = f"Calories: {nutri.get('calories', 'N/A')}\n"
                        info += f"Carbs: {nutri.get('carbs', 'N/A')}\n"
                        info += f"Fiber: {nutri.get('fiber', 'N/A')}\n"
                        if 'vitamin_c' in nutri: info += f"Vit C: {nutri['vitamin_c']}\n"
                        if 'potassium' in nutri: info += f"Potassium: {nutri['potassium']}\n"
                        self.nutrition_panel.delete("0.0", "end")
                        self.nutrition_panel.insert("0.0", f"Nutritional Info:\n{info}")
                    else:
                        self.nutrition_panel.delete("0.0", "end")
                        self.nutrition_panel.insert("0.0", "Nutritional Info:\nNo data available")
                else:
                    # Found by YOLO but NOT in database - don't show info
                    self.res_name_label.configure(text="Fruit: Unknown (Not in DB)")
                    self.res_cond_label.configure(text="Condition: -")
                    self.nutrition_panel.delete("0.0", "end")
                    self.nutrition_panel.insert("0.0", "Nutritional Info:\nPlease train this fruit first")
            elif self.mode == "recognize":
                self.res_name_label.configure(text="Fruit: Searching...")
                self.res_cond_label.configure(text="Condition: -")

            # Convert to PhotoImage
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = img
            self.video_label.configure(image=img)
            
        self.after(10, self.update_video)

if __name__ == "__main__":
    app = App()
    app.mainloop()
