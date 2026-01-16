module.exports = {
  apps: [
    {
      name: 'fastapi-upload-server',
      // *** เปลี่ยน python.exe เป็น pythonw.exe ***
      script: 'C:/Users/natna/Desktop/OMG/CAMERA_AI/venv_311/Scripts/pythonw.exe',

      args: '-m uvicorn main:app --host 0.0.0.0 --port 8000',

      watch: false,
      exec_mode: 'fork',
      instances: 1,
    },
  ],
};