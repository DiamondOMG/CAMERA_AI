"""
Wake Word Service - "Jarvis" (Powered by OpenWakeWord)
Offline 100% | No API Key | Free

Require:
pip install openwakeword pyaudio numpy requests
"""

import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
import requests
import time
import os
# โหลดโมเดล python -c "import openwakeword; openwakeword.utils.download_models()"
# --- ตั้งค่า ---
JARVIS_API_URL = "http://localhost:3000/api/trigger"
CHUNK_SIZE = 1280
MODEL_NAME = "hey_jarvis" # มีให้เลือก: hey_jarvis, alexa, hey_mycroft, etc.
THRESHOLD = 0.5 # เพิ่มขึ้นนิดหน่อยเพื่อความแม่นยำ (0.0 - 1.0)
COOLDOWN_SECONDS = 4 # เพิ่มเวลาพักหน่อยครับ

def trigger_jarvis():
    """ยิง API ไปปลุก Jarvis"""
    try:
        # ยิงคำสั่งเดียว: ปลุก + ทักทาย
        print("🚀 Sending Wake & Greet command...")
        response = requests.post(
            JARVIS_API_URL,
            json={
                "action": "start", 
            },
            timeout=1
        )
        
        if response.status_code == 200:
             print("✅ Command sent!")
        else:
            print(f"⚠️ API error: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Connection error: {e}")

def main():
    print("=" * 50)
    print("🔓 OpenWakeWord Service (Offline & Free)")
    print("=" * 50)
    
    # โหลด Model
    print(f"📥 Loading model: {MODEL_NAME}...")
    try:
        # โหลด OpenWakeWord Model (ระบุ framework='onnx')
        owwModel = Model(
            wakeword_models=[MODEL_NAME], 
            inference_framework="onnx"
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("✅ Model loaded!")

    # เปิดไมโครโฟน
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print(f"\n👂 Listening for '{MODEL_NAME}'...")
    print("   (Note: ลองพูด 'Hey Jarvis' ชัดๆ)")
    
    last_trigger = 0

    try:
        while True:
            # อ่านข้อมูลเสียง
            audio_data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)
            
            # ส่งเข้า Model Prediction
            prediction = owwModel.predict(audio_data)
            
            # prediction เป็น dict เช่น {'hey_jarvis': 0.002, ...}
            score = prediction[MODEL_NAME]
            
            if score > THRESHOLD:
                now = time.time()
                
                # เช็ค Cooldown ก่อนค่อยพิมพ์/ประมวลผล
                if now - last_trigger > COOLDOWN_SECONDS:
                    print(f"⚡ Wake Word Detected! (Score: {score:.3f})")
                    trigger_jarvis()
                    last_trigger = now
                # ถ้าไม่พ้น Cooldown จะเงียบไว้ ไม่พิมพ์รัวๆ แล้วครับ

    except KeyboardInterrupt:
        print("\n⏹️  Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
