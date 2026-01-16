from fastapi import FastAPI, Request, Header, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime

# --- การตั้งค่าเริ่มต้น ---
app = FastAPI()

# กำหนดไดเรกทอรีสำหรับบันทึกไฟล์ และสร้างถ้ายังไม่มี
SAVE_DIR = os.path.join("image", "IMAGE_001")
os.makedirs(SAVE_DIR, exist_ok=True)

# --- ฟังก์ชันหลักสำหรับอัปโหลด ---

@app.post("/upload_binary")
async def upload_image(
    request: Request, 
    # อ่าน Custom Header ที่ส่งมาจาก ESP32
    x_file_name: str = Header(None) 
):
    try:
        # 1. อ่านข้อมูล Raw Bytes จาก Body
        file_bytes = await request.body()
        
        if not file_bytes:
            return JSONResponse({"status": "error", "message": "No file data received"}, status_code=400)

        # 2. กำหนดชื่อไฟล์
        if x_file_name:
            # ใช้ชื่อไฟล์ที่ส่งมาจาก Header
            # ใช้ os.path.basename เพื่อป้องกัน Path Traversal
            base_filename = os.path.basename(x_file_name)
            filename = base_filename 
        else:
            # หากไม่มี Header แนบมา ให้สร้างชื่อไฟล์ตามวันที่และเวลา
            filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".dat" 
            # ใช้นามสกุล .dat หรือ .bin หากไม่ทราบประเภทไฟล์ที่แน่ชัด
        
        # รับค่า folder จาก query parameter (ถ้าไม่ส่งมาจะใช้ค่าเริ่มต้น)
        folder = request.query_params.get("folder", "IMAGE_001")
        
        # สร้าง path สำหรับบันทึกไฟล์
        current_save_dir = os.path.join("image", folder)
        os.makedirs(current_save_dir, exist_ok=True)

        file_path = os.path.join(current_save_dir, filename)

        # 3. บันทึกไฟล์
        with open(file_path, "wb") as buffer:
            buffer.write(file_bytes)

        # 4. ตอบกลับสำเร็จ
        return JSONResponse({"status": "ok", "file_saved": filename})
    
    except Exception as e:
        # ดักจับข้อผิดพลาดทั่วไป
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/upload_form_data")
async def upload_image_form_data(
    file: UploadFile = File(...),
    folder: str = "IMAGE_001"
):
    try:
        # สร้าง path สำหรับบันทึกไฟล์
        current_save_dir = os.path.join("image", folder)
        os.makedirs(current_save_dir, exist_ok=True)
        
        filename = file.filename
        if not filename:
             filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".dat"
             
        file_path = os.path.join(current_save_dir, filename)
        
        # บันทึกไฟล์
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        return JSONResponse({"status": "ok", "file_saved": filename})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# --- รัน Uvicorn ---

if __name__ == "__main__":
    # ใช้ uvicorn.run แทนการรันด้วยคำสั่งภายนอก (เมื่อรันไฟล์โดยตรง)
    uvicorn.run(app, host="0.0.0.0", port=8000)