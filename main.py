from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime

app = FastAPI()

SAVE_DIR = "IMAGE_001"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext
    file_path = os.path.join(SAVE_DIR, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return JSONResponse({"status": "ok", "file_saved": filename})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
