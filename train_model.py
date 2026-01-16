import face_recognition
import cv2
import os
import pickle
import numpy as np
from pathlib import Path

# --- ตั้งค่า ---
TRAIN_DIR = "train_images"
DB_PATH = "test/output/face_database.pkl"
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MODEL = "hog"  # หรือ "cnn" ถ้ามี GPU (hog เร็วกว่าแต่แม่นน้อยกว่านิดหน่อย)

def train():
    print(f"🚀 เริ่มต้นการ Train Model จากโฟลเดอร์: {TRAIN_DIR}")
    
    # 1. เตรียมตัวแปรสำหรับเก็บข้อมูล
    known_face_encodings = []
    known_face_ids = []
    known_face_names = {}
    
    # เริ่มต้น ID ที่ 1
    current_face_id = 1
    
    # ตรวจสอบว่าโฟลเดอร์ train มีจริงไหม
    train_path = Path(TRAIN_DIR)
    if not train_path.exists():
        print(f"❌ ไม่พบโฟลเดอร์ {TRAIN_DIR}")
        return

    # 2. วนลูปอ่านชื่อโฟลเดอร์ (ชื่อคน)
    # เรียงตามชื่อเพื่อความแน่นอนของ ID
    person_dirs = sorted([d for d in train_path.iterdir() if d.is_dir()])
    
    if not person_dirs:
        print("⚠️ ไม่พบโฟลเดอร์ชื่อคนใน train_images เลย")
        return

    print(f"found {len(person_dirs)} people folders: {[d.name for d in person_dirs]}\n")

    for person_dir in person_dirs:
        person_name = person_dir.name
        print(f"👤 กำลังประมวลผล: {person_name} (ID: {current_face_id})")
        
        # เก็บชื่อคู่กับ ID
        known_face_names[current_face_id] = person_name
        
        # หาไฟล์รูปในโฟลเดอร์นั้น
        image_files = [
            f for f in person_dir.iterdir() 
            if f.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        
        count_added = 0
        
        for img_path in image_files:
            try:
                # โหลดภาพผ่าน face_recognition (มันจัดการโหลดเป็น RGB ให้เอง)
                image = face_recognition.load_image_file(str(img_path))
                
                # หาตำแหน่งใบหน้า (เพื่อให้ชัวร์ว่ามีหน้าคน)
                face_locations = face_recognition.face_locations(image, model=MODEL)
                
                if not face_locations:
                    print(f"  Warning: ไม่พบใบหน้าในภาพ {img_path.name}")
                    continue
                
                # แปลงเป็น Vector (Encoding)
                # ปกติ 1 รูปควรมี 1 คน ถ้ามีหลายคน อาจจะต้องเลือกหน้าทีใหญ่สุด
                # แต่ที่นี่เราสมมติว่า crop มาดีแล้ว หรือเอาหน้าแรกที่เจอ
                encodings = face_recognition.face_encodings(image, face_locations)
                
                if len(encodings) > 0:
                    # เอาหน้าแรกที่เจอ (ปกติควรมีหน้าเดียวต่อไฟล์เทรน)
                    encoding = encodings[0]
                    
                    known_face_encodings.append(encoding)
                    known_face_ids.append(current_face_id)
                    count_added += 1
                    print(f"  ✓ เพิ่มข้อมูลจาก {img_path.name}")
                    
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")
        
        print(f"  -> สรุป: เพิ่ม {count_added} ใบหน้าสำหรับ {person_name}")
        print("-" * 30)
        
        # ขยับไปคนถัดไป (ถ้าเจอน่าอย่างน้อย 1 รูป ค่อยเลื่อน ID ก็ได้)
        # แต่เพื่อความง่าย ยึดตามโฟลเดอร์เลย
        current_face_id += 1

    # 3. บันทึกลงไฟล์
    data = {
        'encodings': known_face_encodings,
        'ids': known_face_ids,
        'names': known_face_names,
        'next_id': current_face_id,
        'tolerance': 0.6
    }
    
    # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with open(DB_PATH, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"\n✅ บันทึกข้อมูลเสร็จสิ้นที่: {DB_PATH}")
    print(f"   จำนวน ID ทั้งหมด: {len(known_face_names)}")
    print(f"   จำนวนตัวอย่างใบหน้าทั้งหมด: {len(known_face_ids)}")

if __name__ == "__main__":
    train()
