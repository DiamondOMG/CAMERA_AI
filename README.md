# CAMERA_AI
source venv/Scripts/activate
source venv_311/Scripts/activate

uvicorn main:app --host 0.0.0.0 --port 8000

ngrok http 8000

curl -X POST -H "Content-Type: application/octet-stream" -H "X-File-Name: image_from_curl_123.jpg" --data-binary "@picture_test.jpg" https://a79d668cc454.ngrok-free.app/upload
