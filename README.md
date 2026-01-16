
# CAMERA_AI
# Devmode
source venv_311/Scripts/activate

ถ้าต้องการเปิดเซิฟเวอร์
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

ngrok http 8000

curl -X POST -H "Content-Type: application/octet-stream" -H "X-File-Name: image_from_curl_123.jpg" --data-binary "@picture_test.jpg" https://a79d668cc454.ngrok-free.app/upload

# Production
pm2 kill
pm2 start pm2.config.js