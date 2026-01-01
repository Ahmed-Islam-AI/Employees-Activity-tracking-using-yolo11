# System Architecture - Enhanced Employee Tracking

## 🏗️ System Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       VIDEO INPUT                            │
│            (MP4, AVI, MOV, or Webcam Feed)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │   Frame-by-Frame Processing  │
            └─────────────┬───────────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│   YOLO DETECTION      │   │   POSE ESTIMATION     │
│   (YOLOv4-tiny)       │   │   (OpenPose COCO)     │
├───────────────────────┤   ├───────────────────────┤
│ • Detect Persons      │   │ • Detect 18 Keypoints │
│ • Detect Cell Phones  │   │ • Body Pose Analysis  │
│ • Get Bounding Boxes  │   │ • Joint Positions     │
└───────────┬───────────┘   └───────────┬───────────┘
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │     ANALYSIS ENGINE          │
            ├─────────────────────────────┤
            │                             │
            │  1. POSTURE ANALYSIS        │
            │     - Calculate knee-hip    │
            │       ratio                 │
            │     - Determine Sitting/    │
            │       Standing              │
            │                             │
            │  2. PROXIMITY DETECTION     │
            │     - Measure person-phone  │
            │       distance              │
            │     - Flag phone usage      │
            │                             │
            │  3. ACTIVITY CLASSIFICATION │
            │     - Working (no phone)    │
            │     - Using Phone           │
            │                             │
            └──────────────┬──────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │    VISUALIZATION ENGINE       │
            ├──────────────────────────────┤
            │ • Draw bounding boxes        │
            │ • Display keypoints          │
            │ • Add labels (posture,       │
            │   activity, confidence)      │
            │ • Color-code status          │
            └──────────────┬───────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │         OUTPUT                │
            ├──────────────────────────────┤
            │ ✓ Annotated Video (MP4)      │
            │ ✓ Frame Snapshots (JPG)      │
            │ ✓ Detailed Logs (TXT)        │
            │ ✓ Real-time Display          │
            └──────────────────────────────┘
```

## 🔧 Component Details

### 1️⃣ Video Input Module

- **Supported Formats**: MP4, AVI, MOV, MKV, WebM
- **Webcam Support**: Yes (device index 0, 1, etc.)
- **Frame Rate**: Maintains original video FPS
- **Resolution**: Auto-resized to 800px width for processing

### 2️⃣ YOLO Detection (Object Detection)

- **Model**: YOLOv4-tiny (Darknet)
- **Classes Detected**: Person (0), Cell Phone (67)
- **Input Size**: 416 x 416 pixels
- **Output**: Bounding boxes with confidence scores
- **Speed**: ~30 FPS on CPU
- **Accuracy**: Configurable threshold (default: 0.5)

### 3️⃣ Pose Estimation (OpenPose)

- **Model**: OpenPose COCO (Caffe)
- **Keypoints**: 18 body points
  - Head: Nose, Eyes (L/R), Ears (L/R)
  - Torso: Neck, Shoulders (L/R)
  - Arms: Elbows (L/R), Wrists (L/R)
  - Legs: Hips (L/R), Knees (L/R), Ankles (L/R)
- **Input Size**: 368 x 368 pixels per person
- **Output**: Keypoint coordinates + confidence
- **Processing**: Per-person ROI (Region of Interest)

### 4️⃣ Analysis Algorithms

#### Posture Analysis

```python
torso_length = avg_hip_y - neck_y
knee_hip_distance = avg_knee_y - avg_hip_y
ratio = knee_hip_distance / torso_length

if ratio < 0.5:
    posture = "Sitting"
elif ratio > 0.8:
    posture = "Standing"
else:
    posture = "Sitting/Bending"
```

#### Phone Usage Detection

```python
person_center = (x + w/2, y + h/2)
phone_center = (phone_x + phone_w/2, phone_y + phone_h/2)
distance = euclidean_distance(person_center, phone_center)

if distance < person_radius * 1.5:
    using_phone = True
```

### 5️⃣ Visualization Engine

- **Bounding Boxes**:
  - Green: Working
  - Red: Using phone
  - Magenta: Phone objects
- **Keypoints**: Yellow circles (3px radius)
- **Skeleton Lines**: Connect keypoint pairs
- **Labels**: Posture, Activity, Confidence
- **Overlay Info**: Frame count, timestamp, person count

### 6️⃣ Output Module

- **Video Writer**: OpenCV VideoWriter (MP4V codec)
- **Frame Saver**: Snapshots every 10 seconds
- **Logger**: Timestamped event logging
- **Display**: Real-time preview window

## 📊 Data Flow

```
Input Video → Frames → Detection → Analysis → Annotation → Output
     ↓                                           ↓
  [30 FPS]                                   [Logs]
```

## 💾 Model Files (Auto-downloaded)

| File                         | Size    | Purpose            |
| ---------------------------- | ------- | ------------------ |
| yolov4-tiny.weights          | ~23 MB  | YOLO model weights |
| yolov4-tiny.cfg              | ~6 KB   | YOLO configuration |
| coco.names                   | ~1 KB   | Class labels       |
| pose_deploy_linevec.prototxt | ~30 KB  | OpenPose config    |
| pose_iter_440000.caffemodel  | ~200 MB | OpenPose weights   |

**Total**: ~230 MB (one-time download)

## ⚙️ Configuration Parameters

### Detection Settings

```python
confidence_threshold = 0.5      # Detection confidence (0.0-1.0)
nms_threshold = 0.4             # Non-max suppression
```

### Posture Thresholds

```python
sitting_threshold = 0.5         # Knee-hip ratio for sitting
standing_threshold = 0.8        # Knee-hip ratio for standing
```

### Proximity Settings

```python
phone_proximity_factor = 1.5    # Multiplier of person radius
```

### Output Settings

```python
output_fps = original_fps       # Match input video FPS
snapshot_interval = 10          # Seconds between snapshots
```

## 🔄 Processing Pipeline

1. **Read Frame** → Single frame from video
2. **Resize** → 800px width (maintain aspect ratio)
3. **YOLO Forward** → Detect persons & phones
4. **For Each Person**:
   - Extract person ROI
   - OpenPose forward pass
   - Extract 18 keypoints
   - Calculate posture
   - Check phone proximity
   - Classify activity
5. **Annotate Frame** → Draw boxes, keypoints, labels
6. **Write Output** → Save to video file
7. **Display** → Show in window
8. **Log Events** → Record to log file

## 🎯 Performance Characteristics

| Metric            | Value                        |
| ----------------- | ---------------------------- |
| CPU Usage         | 70-90% (per core)            |
| Memory            | ~500 MB                      |
| Processing Speed  | ~2-5 min per 1-min video     |
| Accuracy (Person) | ~90% (well-lit scenes)       |
| Accuracy (Phone)  | ~70% (depends on phone size) |
| Accuracy (Pose)   | ~85% (full body visible)     |

## 🔐 Privacy & Security

- **Local Processing**: All computation on your machine
- **No Cloud**: No data sent to external servers
- **Offline Capable**: Works without internet (after model download)
- **Data Retention**: User controls output files
- **Encryption**: Consider encrypting output videos

## 🚀 Optimization Tips

1. **Reduce Video Resolution**: Lower quality = faster processing
2. **Increase Confidence**: Higher threshold = fewer false positives
3. **GPU Acceleration**: Modify to use CUDA (requires GPU)
4. **Batch Processing**: Process multiple videos overnight
5. **ROI Limitation**: Only analyze specific areas

## 🔮 Future Enhancements

- [ ] GPU acceleration (CUDA/OpenCL)
- [ ] Multi-person tracking with IDs
- [ ] Time-series analytics
- [ ] Heat maps of activity zones
- [ ] Attention detection (looking at screen)
- [ ] Fatigue/posture quality assessment
- [ ] Export to CSV/database
- [ ] Real-time processing optimization
- [ ] Mobile app viewer
- [ ] Cloud storage integration

---

**Architecture Version**: 1.0  
**Last Updated**: December 29, 2025
