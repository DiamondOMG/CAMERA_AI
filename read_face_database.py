import pickle
import numpy as np

# ระบุ Path ให้ถูกต้อง
db_path = "test/output/face_database.pkl"

try:
    with open(db_path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"--- อ่านข้อมูลจาก {db_path} สำเร็จ ---")
    print(f"Keys ใน Data: {list(data.keys())}")
    
    # ดึงข้อมูลออกมา
    ids = data['ids']
    names = data['names']
    encodings = data['encodings']
    tolerance = data.get('tolerance', 'N/A')
    
    print(f"Tolerance (ความเข้มงวด): {tolerance}")
    print(f"\nพบข้อมูลทั้งหมด {len(ids)} รายการ:\n")
    
    for i, face_id in enumerate(ids):
        name = names.get(face_id, "Unknown")
        encoding = encodings[i]
        
        # แสดง 5 ตัวแรกของ Encoding
        encoding_sample = ", ".join([f"{x:.4f}" for x in encoding[:5]])
        print(f"ID: {face_id} | Name: {name}")
        print(f"   Encoding (first 5): [{encoding_sample}, ...]")
        print("-" * 50)
        
except FileNotFoundError:
    print(f"ไม่พบไฟล์ที่: {db_path}")
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")