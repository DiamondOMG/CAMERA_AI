"""
Camera Service - Central Video Publisher
เปิดกล้องเพียง 1 ครั้ง แล้ว publish frames ผ่าน ZeroMQ
ให้ service อื่นๆ subscribe ได้พร้อมกัน

Required:
pip install opencv-python zmq pyzmq
"""

import cv2
import zmq
import time
import numpy as np
from datetime import datetime

# --- ตั้งค่า ---
CAMERA_INDEX = 0  # กล้องหลัก (0)
ZMQ_PORT = 5555   # Port สำหรับ publish
FPS_TARGET = 30   # ความเร็วที่ต้องการ (frame/sec)

class CameraService:
    def __init__(self, camera_index=CAMERA_INDEX, port=ZMQ_PORT):
        self.camera_index = camera_index
        self.port = port
        self.running = False
        
        # Setup ZeroMQ Publisher
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        
        # Camera
        self.cap = None
        
    def start(self):
        """เริ่มต้น Camera Service"""
        print("=" * 60)
        print("📹 Camera Service - ZeroMQ Publisher")
        print("=" * 60)
        
        # เปิดกล้อง
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ ไม่สามารถเปิดกล้อง index {self.camera_index} ได้")
            return
        
        # ตั้งค่ากล้อง
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ เปิดกล้องสำเร็จ")
        print(f"   📐 Resolution: {width}x{height}")
        print(f"   ⚡ FPS: {actual_fps:.1f}")
        print(f"   🔌 Publishing on: tcp://localhost:{self.port}")
        print(f"\n🚀 กำลัง publish frames...")
        print("   (Subscribers สามารถเชื่อมต่อได้แล้ว)")
        print("   กด Ctrl+C เพื่อหยุด\n")
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️ ไม่สามารถอ่าน frame จากกล้องได้")
                    time.sleep(0.1)
                    continue
                
                # Encode frame เป็น JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # สร้าง metadata
                metadata = {
                    'timestamp': time.time(),
                    'frame_count': frame_count,
                    'width': width,
                    'height': height
                }
                
                # Publish: metadata + frame data
                self.socket.send_json(metadata, zmq.SNDMORE)
                self.socket.send(buffer.tobytes())
                
                frame_count += 1
                
                # แสดง FPS ทุก 5 วินาที
                if frame_count % (FPS_TARGET * 5) == 0:
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Published {frame_count} frames @ {actual_fps:.1f} FPS")
                
                # จำกัด FPS
                time.sleep(1/FPS_TARGET)
                
        except KeyboardInterrupt:
            print("\n⏹️ หยุดการทำงาน...")
        finally:
            self.stop()
    
    def stop(self):
        """ปิด Camera Service"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.socket.close()
        self.context.term()
        print("✅ ปิดกล้องและ socket เรียบร้อย")

if __name__ == "__main__":
    service = CameraService()
    service.start()
