"""
Webcam Face Recognition Service (ZeroMQ Subscriber)
รับ video frames จาก camera_service.py แล้วทำ Face Recognition

Required:
pip install zmq pyzmq opencv-python face_recognition requests
"""

import cv2
import zmq
import time
import requests
import face_recognition
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# --- ตั้งค่า ---
DB_PATH = "test/output/face_database.pkl"
TOLERANCE = 0.45  # ค่า distance ที่ยอมรับ
MODEL = "hog"    # hog สำหรับ CPU
PROCESS_EVERY_N_FRAMES = 30
FRAME_RESIZE_SCALE = 0.25

# --- ZMQ ---
ZMQ_HOST = "localhost"
ZMQ_PORT = 5555

# --- Jarvis Integration ---
JARVIS_API_URL = "http://localhost:3000/api/trigger"
JARVIS_ENABLED = True
GREETING_COOLDOWN = 60

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

class WebcamServiceZMQ:
    def __init__(self):
        self.db = FaceDatabase(DB_PATH)
        self.last_greeted = {}
        self.frame_count = 0
        
        # Setup ZMQ Subscriber
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe ทุก topic

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
                message = f"กล่าวทักทาย {display_name} หน่อย พร้อมเอ่ยชื่อ"
            else:
                message = "ทักทายคนแปลกหน้าหน่อย พร้อมถามชื่อ"
            
            print(f"🔔 แจ้งเตือน Jarvis: {message}")
            requests.post(
                JARVIS_API_URL,
                json={"action": "wakeAndGreet", "message": message},
                timeout=5
            )
            self.last_greeted[greeting_key] = now
        except Exception:
            pass  # เงียบไว้ถ้าต่อ Jarvis ไม่สำเร็จ

    def run(self):
        print("=" * 60)
        print("👤 Face Recognition Service (ZMQ Subscriber)")
        print("=" * 60)
        print(f"🔌 Connecting to camera service at tcp://{ZMQ_HOST}:{ZMQ_PORT}")
        print("⏳ รอรับ frames จาก camera_service...")
        print("กด Ctrl+C เพื่อหยุด\n")

        try:
            while True:
                # รับข้อมูลจาก Publisher
                try:
                    metadata = self.socket.recv_json(zmq.NOBLOCK)
                    frame_data = self.socket.recv()
                except zmq.Again:
                    time.sleep(0.01)
                    continue
                
                # Decode frame
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue

                self.frame_count += 1
                
                # ประมวลผลเฉพาะเฟรมที่กำหนด
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
            self.socket.close()
            self.context.term()
            print("✅ ปิด subscriber เรียบร้อย")

if __name__ == "__main__":
    service = WebcamServiceZMQ()
    service.run()
