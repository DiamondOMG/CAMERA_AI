"""
Webcam Face Recognition Service
จับภาพจากเว็บแคมและเทียบกับฐานข้อมูลใบหน้าแบบ Real-time
"""

import cv2
import time
import requests
import face_recognition
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# --- ตั้งค่า ---
DB_PATH = "test/output/face_database.pkl"
TOLERANCE = 0.6  # ค่า distance ที่ยอมรับ (ยิ่งต่ำยิ่งเข้มงวด)
MODEL = "hog"    # hog สำหรับ CPU, cnn สำหรับ GPU (ถ้ามี)
PROCESS_EVERY_N_FRAMES = 30  # ประมวลผลทุกๆ 30 เฟรม (ลดการใช้ CPU)
FRAME_RESIZE_SCALE = 0.25    # ย่อขนาดเฟรมตอนประมวลผล (0.25 = 1/4)

# --- Jarvis Integration ---
JARVIS_API_URL = "http://localhost:3000/api/trigger"
JARVIS_ENABLED = True
GREETING_COOLDOWN = 60  # วินาที

# --- Name Mapping ---
NAME_MAPPING = {
    "mond": "ม่อน",
    "neab": "เนี๊ยบ",
    "p_hok": "พี่หก",
    "p_nus": "พี่นัส",
    "p_ohm": "พี่โอม",
}

class FaceDatabase:
    """โหลดและจัดการฐานข้อมูลใบหน้า"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.encodings = []
        self.ids = []
        self.names = {}
        self.load()
    
    def load(self):
        if not Path(self.db_path).exists():
            print(f"⚠️  ไม่พบฐานข้อมูลที่ {self.db_path}")
            return False
        
        with open(self.db_path, 'rb') as f:
            data = pickle.load(f)
        
        self.encodings = data['encodings']
        self.ids = data['ids']
        self.names = data['names']
        print(f"✅ โหลดฐานข้อมูลสำเร็จ: {len(self.encodings)} ตัวอย่าง")
        return True

    def find_match(self, face_encoding):
        if not self.encodings:
            return None, None
        
        distances = face_recognition.face_distance(self.encodings, face_encoding)
        best_match_idx = distances.argmin()
        best_distance = distances[best_match_idx]
        
        if best_distance <= TOLERANCE:
            face_id = self.ids[best_match_idx]
            name = self.names.get(face_id, f"ID_{face_id}")
            return name, best_distance
        
        return None, best_distance

class WebcamService:
    def __init__(self):
        self.db = FaceDatabase(DB_PATH)
        self.last_greeted = {}
        self.frame_count = 0

    def notify_jarvis(self, name: str | None):
        if not JARVIS_ENABLED:
            return
        
        display_name = NAME_MAPPING.get(name, name) if name else None
        greeting_key = name if name else "unknown"
        
        now = time.time()
        if now - self.last_greeted.get(greeting_key, 0) < GREETING_COOLDOWN:
            return
        
        try:
            if display_name:
                message = f"ผมเห็นคุณ {display_name} ผ่านเว็บแคมครับ ทักทายเขาหน่อย"
            else:
                message = "มีคนแปลกหน้าอยู่หน้ากล้องครับ ทักทายเขาหน่อย"
            
            print(f"🔔 แจ้งเตือน Jarvis: {message}")
            requests.post(
                JARVIS_API_URL,
                json={"action": "wakeAndGreet", "message": message},
                timeout=5
            )
            self.last_greeted[greeting_key] = now
        except Exception as e:
            # print(f"⚠️ Jarvis error: {e}")
            pass # เงียบไว้ถ้าต่อ Jarvis ไม่สำเร็จ แต่ทำงานต่อได้

    def run(self):
        # เปิดกล้อง
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("❌ ไม่สามารถเปิด Webcam ได้")
            return

        print("🚀 Webcam Service (Headless Mode) เริ่มทำงานแล้ว...")
        print("ทำงานแบบเบื้องหลัง (ไม่มีหน้าต่าง) ประหยัด CPU...")
        print("กด Ctrl+C เพื่อหยุด")

        try:
            while True:
                # อ่านเฟรมจากกล้อง
                ret, frame = video_capture.read()
                if not ret:
                    break

                self.frame_count += 1
                
                # ประมวลผลเฉพาะเฟรมที่กำหนด (ทุกๆ 30 เฟรม หรือประมาณ 1 วินาที)
                if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    # ย่อขนาดเฟรมเพื่อความเร็ว
                    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
                    
                    # แปลง BGR เป็น RGB
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # หาใบหน้า
                    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    if face_encodings:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 👤 พบใบหน้า {len(face_encodings)} ใบหน้า")
                    
                    for face_encoding in face_encodings:
                        name, distance = self.db.find_match(face_encoding)
                        
                        # แจ้งเตือน Jarvis
                        self.notify_jarvis(name)

        except KeyboardInterrupt:
            print("\n⏹️ หยุดการทำงาน...")
        finally:
            # ปล่อยกล้อง
            video_capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    service = WebcamService()
    service.run()
