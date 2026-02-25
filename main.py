import tkinter as tk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from utils import Detector

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Fruit Recognition System - Production")
        self.geometry("1000x600")

        self.detector = Detector()
        self.cap = cv2.VideoCapture(0)
        
        # UI Layout
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Video Section
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")

        # Info Section
        self.info_panel = ctk.CTkFrame(self)
        self.info_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        ctk.CTkLabel(self.info_panel, text="Detection Results", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=30)
        
        self.res_name_label = ctk.CTkLabel(self.info_panel, text="Fruit: Searching...", font=ctk.CTkFont(size=18))
        self.res_name_label.pack(pady=20)
        
        self.res_cond_label = ctk.CTkLabel(self.info_panel, text="Condition: -", font=ctk.CTkFont(size=18))
        self.res_cond_label.pack(pady=10)
        
        ctk.CTkLabel(self.info_panel, text="Nutrition Info:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(20, 5))
        self.nutrition_text = ctk.CTkLabel(self.info_panel, text="", font=ctk.CTkFont(size=14), justify="left")
        self.nutrition_text.pack(pady=5)
        
        ctk.CTkLabel(self.info_panel, text="System: Active", text_color="green").pack(side="bottom", pady=20)

        self.update_video()

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
                    # Use Green box for recognition
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{d['name']}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    found_fruit = d
                    break # Focus on the first detected fruit for the side panel

            if found_fruit:
                name = found_fruit['name']
                # Check database
                if name.lower() in self.detector.db:
                    condition = self.detector.db[name.lower()]['condition']
                    display_name = name
                    
                    self.res_name_label.configure(text=f"Fruit: {display_name.capitalize()}")
                    self.res_cond_label.configure(text=f"Condition: {condition}")
                    
                    # Show nutrition info
                    nutri = self.detector.get_nutrition(display_name)
                    if nutri:
                        info = f"Calories: {nutri.get('calories', 'N/A')}\n"
                        info += f"Carbs: {nutri.get('carbs', 'N/A')}\n"
                        info += f"Fiber: {nutri.get('fiber', 'N/A')}"
                        self.nutrition_text.configure(text=info)
                    else:
                        self.nutrition_text.configure(text="No data available")
                else:
                    self.res_name_label.configure(text="Fruit: Unknown (Not in DB)")
                    self.res_cond_label.configure(text="Condition: -")
                    self.nutrition_text.configure(text="Please train this fruit first")
            else:
                self.res_name_label.configure(text="Fruit: Searching...")
                self.res_cond_label.configure(text="Condition: -")
                self.nutrition_text.configure(text="")

            # Convert to PhotoImage for TK
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = img
            self.video_label.configure(image=img)
            
        self.after(10, self.update_video)

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
