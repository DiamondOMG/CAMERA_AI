# System Prompt – Image Analytics from ESP32-CAM

## Overview

This system analyzes human presence from ESP32-CAM devices.  
It consists of three main components:

1. **ESP32-CAM Edge Device**
2. **Server API (FastAPI)**
3. **Image Processing Worker and Database (Results Only)**

The system is designed to process high-frequency image capture efficiently, storing only analysis results in the database while raw images remain in file storage.

---

## 1. ESP32-CAM (Edge Device)

- Detect motion using a PIR sensor.
- Capture images at **5 FPS** (~200ms interval).
- **Image file size:** max **50KB** (compressed JPEG)
- File naming convention: timestamp in milliseconds (e.g., `1764318832186.jpg`).
- Store images locally on SD Card.
- Maintain an upload queue to send images to the server via HTTP POST (`/upload?device=ID`).
- Retry uploads automatically if the network is unavailable.
- **No AI processing is performed on the device.**

---

## 2. Server API (FastAPI)

- Endpoint: `POST /upload?device=DEVICE_ID`
- Receives image files via multipart/form-data.
- Stores images in device-specific folders:
  images/<DEVICE_ID>/<TIMESTAMP>.jpg

- Does not perform image analysis on upload.
- Queues tasks for Worker processing.

---

## 3. Worker – Image Processing Pipeline

Worker reads images from device folders and performs:

### Processing Strategy

- **System supports:** up to **10 devices**
- **Processing mode:** Batch/queue-based (not real-time)
- **Batch size:** 100 images per device per cycle
- Poll each device folder every N seconds (e.g., check folder, process 100 images, move to next device)

### Responsibilities

1. Person detection.
2. Unique person identification (tracking across frames).
3. Dwell time calculation using timestamp sequence.
4. Generate events summarizing key results:
   - `start_ts`: entry time
   - `end_ts`: exit time
   - `dwell_time`: duration
   - `person_id`: unique identifier

### Post-Processing

- Store only analysis results in the database.
- Optionally delete raw images after processing or retain them for a defined period.

---

## 4. Database Design (Results Only)

### `events` table

Stores only summarized events (one event may correspond to many images):
| Column | Description |
|--------------|-------------|
| event_id | Unique event identifier |
| device_id | Device identifier |
| person_id | Unique person ID |
| start_ts | Event start timestamp |
| end_ts | Event end timestamp |
| dwell_time | Duration in seconds |

### `person_summary` table (optional)

Aggregates daily statistics:
| Column | Description |
|---------------------|-------------|
| date | Date of summary |
| device_id | Device identifier |
| unique_person_count | Number of unique persons |
| total_dwell_time | Total dwell time per device |

**Note:** Storing only events ensures database remains lightweight (<200 records/day/device) even with high-frequency image capture.

---

## 5. Data Flow

[ESP32-CAM]
PIR Detect → Capture → Save SD → Upload Queue → POST /upload

[Server]
Receive → Save to /images/DEVICE_ID/ → Queue task for Worker

[Worker]
Poll folders (every N seconds) → Collect 100 images per device → Analyze → Generate events → Save to DB → (optional) delete processed images

---

## 6. Rationale for Not Storing Raw Images in DB

- High-frequency capture (0.2s interval) generates ~432,000 images/day/device.
- Storing all metadata in the database would be resource-intensive.
- Keeping only results (events, dwell times, unique person IDs) drastically reduces storage requirements while maintaining essential analytics.

---

## 7. System Goals

- Efficiently analyze images from multiple ESP32-CAM devices.
- Count unique persons.
- Calculate dwell time.
- Store only results to maintain high performance and scalability.
- Enable seamless integration with analytics dashboards.
