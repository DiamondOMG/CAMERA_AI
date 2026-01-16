# System Prompt – ESP32-CAM Face Recognition System

## Overview
ระบบตรวจจับและจดจำใบหน้า (Face Recognition) จากกล้อง ESP32-CAM โดยเน้นการทำงานแบบแยกส่วน (Microservices style) ระหว่างส่วนรับภาพและส่วนประมวลผล

### System Architecture
1. **Edge Device (ESP32-CAM)**: ถ่ายภาพและส่งมาที่ Server
2. **File Server (FastAPI)**: รับไฟล์ภาพและบันทึกลง Disk
3. **Face Watcher (Service)**: เฝ้าดูโฟลเดอร์ภาพและประมวลผลใบหน้า
4. **Face Database (Pickle)**: ฐานข้อมูลใบหน้าและเวกเตอร์ (Encodings)

---

## 1. Flow การทำงาน

### 1.1 การรับภาพ (Upload)
- **Component**: `main.py` (FastAPI)
- **Endpoint**: `POST /upload`
- **หน้าที่**: 
  - รับข้อมูล Raw Bytes / Multipart จาก ESP32
  - อ่าน Header `x-file-name` เพื่อตั้งชื่อไฟล์ (หรือใช้ Timestamp)
  - บันทึกไฟล์ลงโฟลเดอร์ `image/IMAGE_001/`
  - *ไม่มีการประมวลผลใดๆ ในขั้นตอนนี้เพื่อให้ Response เร็วที่สุด*

### 1.2 การประมวลผล (Processing)
- **Component**: `face_watcher.py` (Watchdog Service)
- **หน้าที่**:
  - เฝ้าดู (Monitor) โฟลเดอร์ `image/IMAGE_001/` ตลอดเวลา
  - เมื่อมีไฟล์ใหม่ -> รอให้ไฟล์เขียนเสร็จ -> อ่านภาพ
  - **Detect Face**: ตรวจจับตำแหน่งใบหน้า
  - **Encode & Match**: แปลงใบหน้าเป็น Vector และเทียบกับ `test/output/face_database.pkl`
  - **Logging**: แสดงผลลัพธ์ทาง Console ว่าเจอใคร (Name) หรือเป็นคนแปลกหน้า (Unknown) โดยดูจากค่า Distance
  - **Tracking**: บันทึกชื่อไฟล์ลง `processed_files.txt` เพื่อป้องกันการประมวลผลซ้ำ

### 1.3 การจัดการฐานข้อมูล (Database Management)
- **Training**: `train_model.py`
  - อ่านรูปจากโฟลเดอร์ `train_images/{PERSON_NAME}/`
  - สร้าง Encoding ของทุกคนใหม่ทั้งหมด (Rebuild)
  - บันทึกลงไฟล์ `.pkl`
- **Inference**: `read_face_database.py` (Utility)
  - อ่านเช็คข้อมูลในไฟล์ฐานข้อมูล

---

## 2. โครงสร้างโฟลเดอร์ (Directory Structure)

```text
CAMERA_AI/
├── main.py                # Web Server (FastAPI)
├── face_watcher.py        # Service เฝ้าดูและประมวลผลภาพ
├── train_model.py         # Script สำหรับ Train หน้าคนใหม่
├── train_images/          # โฟลเดอร์เก็บรูปต้นฉบับสำหรับ Train
│   ├── somchai/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── somying/
├── image/
│   └── IMAGE_001/         # โฟลเดอร์รับรูปจาก ESP32 (Watcher เฝ้าตรงนี้)
├── test/
│   └── output/
│       └── face_database.pkl  # ไฟล์ฐานข้อมูลใบหน้า (Model)
└── processed_files.txt    # list ไฟล์ที่ประมวลผลแล้ว
```

## 3. Configuration

- **Models**: ใช้ `hog` (CPU Friendly) หรือ `cnn` (GPU Needed - แม่นยำกว่า)
- **Tolerance**: ค่าความเหมือน (Distance Threshold)
  - `0.6`: ค่ามาตรฐาน
  - `0.5`: เข้มงวดขึ้น (ลด False Positive)
  - `0.4`: เข้มงวดมาก (เฉพาะหน้าชัดๆ)

## 4. Future Improvements
- เชื่อมต่อ Database จริง (MySQL/PostgreSQL) แทน Text File
- ส่ง Notification (Line/Discord) เมื่อเจอคนแปลกหน้า
- ทำ Dashboard แสดงประวัติการเข้า-ออก
