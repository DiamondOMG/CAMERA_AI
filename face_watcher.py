"""
Face Watcher Service
เฝ้าดูโฟลเดอร์รูปภาพ และตรวจจับใบหน้าเทียบกับฐานข้อมูล
"""

import os
import time
import pickle
import face_recognition
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- ตั้งค่า ---
WATCH_DIR = "image/IMAGE_002"
DB_PATH = "test/output/face_database.pkl"
PROCESSED_FILE = "processed_files.txt"  # เก็บรายชื่อไฟล์ที่ประมวลผลแล้ว
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
TOLERANCE = 0.45  # ค่า distance ที่ยอมรับ (ยิ่งต่ำยิ่งเข้มงวด)
MODEL = "hog"


class FaceDatabase:
    """โหลดและจัดการฐานข้อมูลใบหน้า"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.encodings = []
        self.ids = []
        self.names = {}
        self.tolerance = TOLERANCE
        self.load()
    
    def load(self):
        """โหลดฐานข้อมูลจากไฟล์"""
        if not Path(self.db_path).exists():
            print(f"⚠️  ไม่พบฐานข้อมูลที่ {self.db_path}")
            return False
        
        with open(self.db_path, 'rb') as f:
            data = pickle.load(f)
        
        self.encodings = data['encodings']
        self.ids = data['ids']
        self.names = data['names']
        # self.tolerance = data.get('tolerance', TOLERANCE)
        self.tolerance = TOLERANCE  # ใช้ค่าจาก Config ด้านบน (จะได้ปรับจูนง่ายๆ)
        
        print(f"✅ โหลดฐานข้อมูลสำเร็จ: {len(set(self.ids))} คน, {len(self.encodings)} ตัวอย่าง")
        print(f"   ⚙️  Config Tolerance: {self.tolerance} (ต่ำกว่านี้ถึงจะแมทช์)")
        return True
    
    def find_match(self, face_encoding) -> tuple:
        """
        หาใบหน้าที่ตรงกับ encoding ที่ให้มา
        Returns: (name, distance) หรือ (None, None) ถ้าไม่เจอ
        """
        if not self.encodings:
            return None, None
        
        # คำนวณ distance กับทุก encoding ในฐานข้อมูล
        distances = face_recognition.face_distance(self.encodings, face_encoding)
        
        # หาตัวที่ใกล้ที่สุด
        best_match_idx = distances.argmin()
        best_distance = distances[best_match_idx]
        
        # เช็คว่าผ่าน threshold หรือไม่
        if best_distance <= self.tolerance:
            face_id = self.ids[best_match_idx]
            name = self.names.get(face_id, f"ID_{face_id}")
            return name, best_distance
        
        return None, best_distance


class ProcessedFiles:
    """จัดการรายการไฟล์ที่ประมวลผลแล้ว"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.processed = set()
        self.load()
    
    def load(self):
        """โหลดรายการจากไฟล์"""
        if Path(self.filepath).exists():
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.processed = set(line.strip() for line in f if line.strip())
            print(f"📋 โหลดรายการไฟล์ที่ประมวลผลแล้ว: {len(self.processed)} ไฟล์")
    
    def add(self, filename: str):
        """เพิ่มไฟล์เข้ารายการ"""
        self.processed.add(filename)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(filename + '\n')
    
    def is_processed(self, filename: str) -> bool:
        """เช็คว่าประมวลผลไปแล้วหรือยัง"""
        return filename in self.processed


class ImageHandler(FileSystemEventHandler):
    """จัดการ Event เมื่อมีไฟล์ใหม่"""
    
    def __init__(self, db: FaceDatabase, processed: ProcessedFiles):
        self.db = db
        self.processed = processed
    
    def on_created(self, event):
        """เมื่อมีไฟล์ใหม่ถูกสร้าง"""
        if event.is_directory:
            return
        
        filepath = event.src_path
        self.process_image(filepath)
    
    def process_image(self, filepath: str):
        """ประมวลผลรูปภาพ"""
        filename = os.path.basename(filepath)
        ext = Path(filepath).suffix.lower()
        
        # เช็คนามสกุลไฟล์
        if ext not in ALLOWED_EXTENSIONS:
            return
        
        # เช็คว่าประมวลผลไปแล้วหรือยัง
        if self.processed.is_processed(filename):
            return
        
        # รอให้ไฟล์เขียนเสร็จ (ESP32 อาจส่งมาช้า)
        time.sleep(0.5)
        
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] 📷 พบรูปใหม่: {filename}")
            
            # โหลดรูปและหาใบหน้า
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image, model=MODEL)
            
            if not face_locations:
                print(f"   ❌ ไม่พบใบหน้าในรูป")
                self.processed.add(filename)
                return
            
            print(f"   👤 พบใบหน้า {len(face_locations)} ใบหน้า")
            
            # แปลงเป็น encoding
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # เทียบแต่ละใบหน้ากับฐานข้อมูล
            for i, encoding in enumerate(face_encodings):
                name, distance = self.db.find_match(encoding)
                
                if name:
                    print(f"   ✅ ใบหน้า #{i+1}: {name} (distance: {distance:.4f})")
                else:
                    print(f"   ❓ ใบหน้า #{i+1}: Unknown (distance: {distance:.4f})")
            
            # บันทึกว่าประมวลผลแล้ว
            self.processed.add(filename)
            
        except Exception as e:
            print(f"   ⚠️  Error processing {filename}: {e}")


def scan_existing_files(watch_dir: str, handler: ImageHandler):
    """สแกนไฟล์ที่มีอยู่แล้วในโฟลเดอร์"""
    print(f"\n🔍 สแกนไฟล์ที่มีอยู่ใน {watch_dir}...")
    
    watch_path = Path(watch_dir)
    if not watch_path.exists():
        print(f"⚠️  โฟลเดอร์ {watch_dir} ไม่มีอยู่")
        return
    
    count = 0
    for ext in ALLOWED_EXTENSIONS:
        for filepath in watch_path.glob(f"*{ext}"):
            handler.process_image(str(filepath))
            count += 1
    
    print(f"✅ สแกนเสร็จสิ้น: {count} ไฟล์")


def main():
    print("=" * 50)
    print("🚀 Face Watcher Service")
    print("=" * 50)
    
    # โหลดฐานข้อมูลใบหน้า
    db = FaceDatabase(DB_PATH)
    
    # โหลดรายการไฟล์ที่ประมวลผลแล้ว
    processed = ProcessedFiles(PROCESSED_FILE)
    
    # สร้าง Event Handler
    handler = ImageHandler(db, processed)
    
    # สแกนไฟล์ที่มีอยู่แล้ว (ถ้ายังไม่เคยประมวลผล)
    scan_existing_files(WATCH_DIR, handler)
    
    # สร้าง Observer สำหรับเฝ้าดูโฟลเดอร์
    observer = Observer()
    
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(WATCH_DIR, exist_ok=True)
    
    observer.schedule(handler, WATCH_DIR, recursive=False)
    observer.start()
    
    print(f"\n👁️  กำลังเฝ้าดูโฟลเดอร์: {WATCH_DIR}")
    print("   กด Ctrl+C เพื่อหยุด\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  หยุดการทำงาน...")
        observer.stop()
    
    observer.join()
    print("👋 Goodbye!")


if __name__ == "__main__":
    main()
