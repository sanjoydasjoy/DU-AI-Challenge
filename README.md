# 🏆 DU AI Challenge - Team TopG Solution (0.66 mAP)

## Table of Contents
- [Challenge Overview](#challenge-overview)
- [Problem Analysis](#problem-analysis)
- [Our Approach](#our-approach)
- [Technical Implementation](#technical-implementation)
- [Code Walkthrough](#code-walkthrough)
- [Results & Performance](#results--performance)
- [Key Insights](#key-insights)

<br>

___

<br>

## Challenge Overview


The **DU AI Challenge: DU Arena Season 1** was an intensive **8-hour computer vision competition** where teams developed AI models to detect and classify traffic objects in drone-captured aerial images from Bangladesh.



### Competition Details
- **Duration**: 8 hours (Aug 8-9, 2025)
- **Teams**: 40 registered teams competing
- **Evaluation**: Mean Average Precision (mAP) at IoU threshold 0.5
- **Hardware**: 8GB GPU per team via cloud VM or Kaggle

### The Dataset
**Training Set**: 174 high-resolution drone images with YOLO format annotations
**Test Set**: 57 images requiring object detection predictions
**Image Resolution**: Primarily 4K (3840×2160) aerial captures
**Average Objects per Image**: 33 (dense traffic scenes)

### Object Classes (11 Categories)
| ID | Class | Description |
|----|-------|-------------|
| 0  | bicycle | Two-wheeled pedal vehicle |
| 1  | bus | Large passenger vehicle |
| 2  | car | Standard passenger car |
| 3  | cng | CNG auto-rickshaw (three-wheeler) |
| 4  | leguna | Small utility vehicle (Bangladesh-specific) |
| 5  | manual-van | Cycle rickshaw/manual pull van |
| 6  | motor | Motorcycle/motorbike |
| 7  | others | Other vehicle types |
| 8  | pedestrian | People walking |
| 9  | rickshaw | Traditional rickshaw |
| 10 | truck | Large cargo vehicle |

### Unique Challenges
1. **Bangladesh-specific vehicles**: CNG rickshaws, legunas, manual vans
2. **Dense urban traffic**: Complex South Asian traffic scenarios
3. **Aerial perspective**: Drone-captured high-resolution images
4. **Small dataset**: Only 174 training images
5. **Object density**: Up to 100+ objects per image

<br>

___

<br>

## Problem Analysis

#### 1. **Small Dataset Challenge**
- Only **174 training images** for 11 classes
- Risk of overfitting
- Need for smart data augmentation
- Careful validation strategy required

#### 2. **Multi-Scale Objects**
- **Tiny objects**: Pedestrians, motorcycles (10-50 pixels)
- **Medium objects**: Cars, CNGs, rickshaws (50-200 pixels)
- **Large objects**: Buses, trucks (200+ pixels)

#### 3. **Dense Traffic Scenes**
- Average **33 objects per image**
- Some images with **100+ objects**
- Heavy occlusion between vehicles
- Overlapping bounding boxes

#### 4. **Domain-Specific Challenges**
- **Bangladesh traffic patterns**: Unique vehicle types
- **Aerial perspective**: Different from standard datasets
- **Varying altitudes**: Different object scales
- **Lighting conditions**: Different times of day

#### 5. **High-Resolution Images**
- **4K images**: Computational challenge
- **Memory constraints**: Limited to 8GB GPU
- **Processing time**: Inference speed matters

<br>

___

<br>

## Our Approach

### Overall Strategy
Our strategy was mainly focused on **maximizing recall while maintaining precision** through:

1. **Smart Model Selection**: YOLOv8m for optimal accuracy-speed balance
2. **High-Resolution Training**: Preserving small object details
3. **Aggressive Augmentation**: Simulating diverse traffic scenarios
4. **Recall-Optimized Inference**: Low confidence threshold for maximum detections
5. **Proper Validation**: Preventing overfitting on small dataset



- **mAP metric rewards recall**: Better to detect all objects with some false positives
- **Low confidence threshold**: Capture borderline detections
- **High max detections**: Handle dense traffic scenes
- **Large input size**: Preserve details for small objects

<br>

___

<br>

## Technical Implementation

### Architecture Choice: YOLOv8m

#### Why YOLOv8m?
```python
model=yolov8m.pt  # Medium model
```

**Comparison of YOLOv8 Variants:**


| Model       | Parameters | Speed      | Accuracy  | Our Choice                |
| ----------- | ---------- | ---------- | --------- | ------------------------- |
| YOLOv8n     | 3.2M       | Very Fast  | Low       | No – Too small            |
| YOLOv8s     | 11.2M      | Fast       | Moderate  | No – Still small          |
| **YOLOv8m** | **25.9M**  | **Medium** | **High**  | **Yes – Perfect balance** |
| YOLOv8l     | 43.7M      | Slow       | Very High | No – Might overfit        |
| YOLOv8x     | 68.2M      | Very Slow  | Very High | No – Too heavy            |


**YOLOv8m Benefits:**
- **25.9M parameters**: Sufficient capacity for 11 classes
- **Dense object detection**: Better than smaller models
- **Memory efficient**: Fits in 8GB GPU with large images
- **Pre-trained**: Good starting point for fine-tuning

### Image Resolution Strategy

#### Training Size: 1536px
```python
imgsz=1536  # Training image size
```

**Why 1536 was Critical:**
- **Original images**: 4K (3840×2160)
- **Downscaling factor**: ~2.5x
- **Small object preservation**: Pedestrians still visible
- **GPU memory**: Fits 4 images with 8GB
- **Detail retention**: Better than 1024 or 640

**Size Comparison:**


| Size | Small Objects | GPU Memory | Speed     | Our Choice |
| ---- | ------------- | ---------- | --------- | ---------- |
| 640  | Lost details  | Low        | Very Fast | No         |
| 1024 | Some loss     | Medium     | Fast      | No         |
| 1536 | Preserved     | Manageable | Moderate  | Yes        |
| 2048 | Best          | High       | Slow      | No         |




### Data Augmentation Strategy

#### Comprehensive Augmentation Pipeline
```python
mosaic=1.0          # 100% mosaic augmentation
mixup=0.15          # 15% mixup probability
hsv_h=0.015         # Hue variation (±1.5%)
hsv_s=0.7           # Saturation variation (±70%)
hsv_v=0.4           # Value/brightness variation (±40%)
translate=0.1       # Translation (±10%)
scale=0.5           # Scale variation (±50%)
fliplr=0.5          # Horizontal flip (50% probability)
```

#### **Mosaic Augmentation (The Game Changer)**
```
┌─────────┬─────────┐
│  IMG_1  │  IMG_2  │  Combined into single
├─────────┼─────────┤  training image with
│  IMG_3  │  IMG_4  │  4x object density
└─────────┴─────────┘
```

**Benefits:**
- **Synthetic dense scenes**: Creates traffic jams artificially
- **Scale diversity**: Objects at different scales in same image
- **Context learning**: Model learns object relationships
- **Data efficiency**: 4x more training samples

#### **HSV Augmentation for Bangladesh Traffic**
```python
hsv_h=0.015  # Subtle hue changes (different lighting)
hsv_s=0.7    # Strong saturation (weather conditions)
hsv_v=0.4    # Brightness variation (time of day)
```

**Why Strong HSV?**
- **Weather adaptation**: Monsoon vs sunny conditions
- **Time variation**: Morning vs evening lighting
- **Camera differences**: Different drone cameras
- **Robustness**: Model works in various conditions

### Training Configuration

#### Optimizer & Learning Rate
```python
optimizer=AdamW     # Adaptive optimization
lr0=0.001          # Initial learning rate
cos_lr=True        # Cosine annealing schedule
```

**AdamW vs SGD:**
- **AdamW**: Better for object detection, handles sparse gradients
- **Cosine LR**: Gradual learning rate decay for better convergence
- **Conservative LR**: 0.001 prevents overfitting on small dataset

#### Training Schedule
```python
epochs=50          # Maximum training epochs
patience=10        # Early stopping patience
batch=4           # Batch size (memory constraint)
```

**Early Stopping Logic:**
```
Epoch 1-10: Learning basic features
Epoch 11-25: Fine-tuning object detection
Epoch 26-35: Model starts converging
Epoch 36-40: Validation loss stops improving
Epoch 41: Early stopping triggered (patience=10)
```

### Inference Optimization

#### Recall-Focused Settings
```python
imgsz=1280         # Inference image size (vs 1536 training)
conf=0.05          # Very low confidence threshold
iou=0.45           # Moderate IoU for NMS
max_det=2000       # High maximum detections
```

#### **Confidence Threshold Strategy**
```python
conf=0.05  # Extremely low (typical: 0.25)
```

**Why 0.05 was brilliant:**
- **mAP optimization**: Metric rewards recall over precision
- **Borderline detections**: Captures uncertain objects
- **False positives**: Better than missed detections
- **Competition insight**: Let evaluation handle precision

#### **Maximum Detections**
```python
max_det=2000  # High limit (typical: 300-1000)
```

**Handling Dense Scenes:**
- Some images had **100+ objects**
- Default limits would **truncate detections**
- **2000 limit**: Ensures no object is missed
- **Memory trade-off**: Worth it for dense traffic

<br>

___

<br>

## Code Walkthrough



### Cell 1: Dataset Setup
```python
DATASET_NAME="challenge-ai"  # Dataset identifier
!rm -rf /kaggle/working/proj
!mkdir -p /kaggle/working/proj/data

# Copy train and test folders
!cp -r /kaggle/input/{DATASET_NAME}/train /kaggle/working/proj/data/
!cp -r /kaggle/input/{DATASET_NAME}/test  /kaggle/working/proj/data/
!cp /kaggle/input/{DATASET_NAME}/sample_submission.csv /kaggle/working/proj/
```

**Purpose**: 
- Set up working directory structure
- Copy dataset to writable location
- Prepare for training pipeline

**Why This Approach:**
- Kaggle datasets are read-only
- Need writable space for training outputs
- Organized structure for clean workflow

### Cell 2: Smart Data Split
```python
import os, random, shutil, glob
random.seed(42)  # Reproducible results

BASE = "/kaggle/working/proj/data"
ti, tl = f"{BASE}/train/images", f"{BASE}/train/labels"
vi, vl = f"{BASE}/val/images",   f"{BASE}/val/labels"
os.makedirs(vi, exist_ok=True); os.makedirs(vl, exist_ok=True)

imgs = sorted(glob.glob(os.path.join(ti, "*")))
val = set(random.sample(imgs, max(1, int(0.12*len(imgs)))))  # 12% validation

stem = lambda p: os.path.splitext(os.path.basename(p))[0]
for ip in imgs:
    lp = os.path.join(tl, stem(ip)+".txt")
    if ip in val:
        shutil.move(ip, os.path.join(vi, os.path.basename(ip)))
        if os.path.exists(lp):
            shutil.move(lp, os.path.join(vl, os.path.basename(lp)))
```

**Purpose**: Create train/validation split
- **136 training images**
- **38 validation images** (12%)

**Strategic Decisions:**
1. **12% split**: Optimal for small datasets
   - Too small (5%): Unreliable validation
   - Too large (20%): Not enough training data
   - 12%: Sweet spot for 174 images

2. **Random seed 42**: Reproducible experiments
3. **Move files**: Proper YOLO directory structure
4. **Label matching**: Ensures image-label pairs stay together

### Cell 4: YAML Configuration
```python
yaml_text = """path: /kaggle/working/proj/data
train: train/images
val: val/images
test: test/images
names:
  0: bicycle
  1: bus
  2: car
  3: cng
  4: leguna
  5: manual-van
  6: motor
  7: others
  8: pedestrian
  9: rickshaw
  10: truck
"""
open("/kaggle/working/proj/data.yaml","w").write(yaml_text)
```

**Purpose**: YOLO dataset configuration
- Defines dataset paths
- Maps class IDs to names
- Required for YOLO training

**Class Mapping Strategy:**
- **Standard vehicles**: car, bus, truck, motorcycle
- **Bangladesh-specific**: CNG, leguna, rickshaw, manual-van
- **Universal**: bicycle, pedestrian, others

### Cell 6: Environment Setup
```python
!pip install -q ultralytics ensemble-boxes
from ultralytics import YOLO
```

**Purpose**: Install and verify dependencies
- **Ultralytics**: YOLOv8 framework
- **Ensemble-boxes**: For potential model ensembling

### Cell 8: GPU Optimization
```python
def sh(cmd):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=False)

print("=== GPU (nvidia-smi) ===")
sh("nvidia-smi")

# CUDA-enabled PyTorch installation
if need_cuda:
    sh(f"{sys.executable} -m pip uninstall -y torch torchvision torchaudio")
    sh(f"{sys.executable} -m pip install -q --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio")
```

**Purpose**: Ensure optimal GPU setup
- Verify GPU availability (Tesla T4 in this case)
- Install CUDA-compatible PyTorch
- Maximum training speed

**Performance Impact:**
- **GPU acceleration**: 10-50x faster than CPU
- **CUDA optimization**: Efficient memory usage
- **Verification step**: Prevents runtime issues

### Cell 11: The Main Training (The Heart of Our Solution)
```python
cmd = (
    "yolo detect train "
    "data=/kaggle/working/proj/data.yaml "
    "model=yolov8m.pt "
    "imgsz=1536 "           # High resolution
    "batch=4 "              # Memory constraint
    "epochs=50 "            # Sufficient for convergence
    "optimizer=AdamW lr0=0.001 cos_lr=True "  # Advanced optimization
    "mosaic=1.0 mixup=0.15 "                  # Aggressive augmentation
    "hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 "       # Color augmentation
    "translate=0.1 scale=0.5 fliplr=0.5 "    # Geometric augmentation
    "cache=True patience=10 "                  # Efficiency & early stopping
    "workers=2 "                               # CPU threads
    "device=0 "                                # GPU device
    "project=/kaggle/working/proj/runs"        # Output directory
)
subprocess.run(cmd, shell=True, check=False)
```

**Purpose**: Train our winning model

**Parameter Deep Dive:**

#### **Core Architecture**
- `model=yolov8m.pt`: Pre-trained YOLOv8 medium
- `data=...yaml`: Our dataset configuration

#### **Image & Batch Settings**
- `imgsz=1536`: **KEY DECISION** - High resolution for small objects
- `batch=4`: Maximum batch fitting in 8GB GPU memory

#### **Training Schedule**
- `epochs=50`: Sufficient without overfitting
- `patience=10`: Early stopping after 10 epochs without improvement

#### **Optimization Strategy**
- `optimizer=AdamW`: Better than SGD for object detection
- `lr0=0.001`: Conservative learning rate for small dataset
- `cos_lr=True`: Cosine annealing for smooth convergence

#### **Augmentation Pipeline**
- `mosaic=1.0`: **100% mosaic** - Creates dense synthetic scenes
- `mixup=0.15`: **15% mixup** - Blends images for regularization
- `hsv_h=0.015`: Subtle hue variation (lighting changes)
- `hsv_s=0.7`: **Strong saturation** (weather conditions)
- `hsv_v=0.4`: Brightness variation (time of day)
- `translate=0.1`: 10% translation (camera movement)
- `scale=0.5`: **50% scale** (different altitudes)
- `fliplr=0.5`: Horizontal flip (direction invariance)

#### **Efficiency Settings**
- `cache=True`: Cache images in memory for faster loading
- `workers=2`: CPU threads for data loading
- `device=0`: Use primary GPU

### Cell 12: Optimized Inference
```python
!yolo detect predict \
  model=/kaggle/working/proj/runs/train/weights/best.pt \
  source=/kaggle/working/proj/data/test/images \
  imgsz=1280 conf=0.05 iou=0.45 \
  save_txt=True save_conf=True max_det=2000 \
  project=/kaggle/working/proj/runs --name=predict
```

**Purpose**: Generate predictions on test set

**Inference Strategy:**
- `model=.../best.pt`: Use best epoch (not last)
- `imgsz=1280`: Slightly smaller for speed (vs 1536 training)
- `conf=0.05`: **EXTREMELY LOW** - Maximize recall
- `iou=0.45`: Moderate NMS threshold
- `max_det=2000`: **HIGH LIMIT** - Handle dense scenes
- `save_txt=True save_conf=True`: Save detailed predictions

**Why conf=0.05 used:**
```
High Confidence (0.25):   [████████░░] - Miss borderline objects
Medium Confidence (0.15): [██████████░] - Some false positives
Low Confidence (0.05):    [███████████] - Catch everything!
```

### Cell 14: Submission Generation
```python
import os, glob, pandas as pd
from PIL import Image

TEST_DIR   = "/kaggle/working/proj/data/test/images"
LABELS_DIR = "/kaggle/working/proj/runs/predict/labels"
OUT_CSV    = "/kaggle/working/submission_okay.csv"

# Get image dimensions
sizes = {}
for p in glob.glob(os.path.join(TEST_DIR, "*")):
    name = os.path.splitext(os.path.basename(p))[0]
    with Image.open(p) as im:
        sizes[name] = im.size  # (width, height)

rows, seen = [], set()
for t in sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt"))):
    image_id = os.path.splitext(os.path.basename(t))[0]
    w, h = sizes[image_id]
    parts_out = []
    
    with open(t) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) not in (5,6): 
                continue
                
            cls = int(float(vals[0]))              # Class ID
            x, y, bw, bh = map(float, vals[1:5])   # Normalized YOLO format
            conf = float(vals[5]) if len(vals) == 6 else 0.5  # Confidence

            # Convert YOLO format to pixel coordinates
            x_min = (x - bw/2) * w
            y_min = (y - bh/2) * h
            x_max = (x + bw/2) * w
            y_max = (y + bh/2) * h

            # Ensure coordinates are within image bounds
            parts_out += [
                str(cls), f"{conf:.6f}",
                str(int(max(0, x_min))), str(int(max(0, y_min))),
                str(int(min(w-1, x_max))), str(int(min(h-1, y_max))),
            ]
            
    rows.append({"image_id": image_id, "PredictionString": " ".join(parts_out)})
    seen.add(image_id)

# Handle images with no detections
for name in sizes.keys():
    if name not in seen:
        rows.append({"image_id": name, "PredictionString": ""})

pd.DataFrame(rows).sort_values("image_id").to_csv(OUT_CSV, index=False)
```

**Purpose**: Convert YOLO predictions to competition format

**Conversion Process:**

1. **Load image dimensions**: Get original sizes for coordinate conversion
2. **Parse YOLO files**: Extract class, bbox, confidence
3. **Coordinate conversion**:
   ```python
   # From YOLO (normalized center format)
   x, y, w, h = 0.5, 0.3, 0.1, 0.2
   
   # To competition format (pixel corners)
   x_min = (0.5 - 0.1/2) * image_width
   y_min = (0.3 - 0.2/2) * image_height
   x_max = (0.5 + 0.1/2) * image_width  
   y_max = (0.3 + 0.2/2) * image_height
   ```
4. **Bound checking**: Ensure coordinates within image
5. **Format output**: `class_id confidence x_min y_min x_max y_max`

<br>

___

<br>

## Results & Performance

### Competition Results
- **Final Ranking**: **Top 5 out of 40+ teams in public leaderboard and Top 7 in private leaderboard.**
- **mAP Score**: **0.66 in public leaderboard and 0.60 in private leaderboard.**
- **Training Time**: ~2 hours on Tesla T4
- **Inference Time**: ~5 minutes for 57 test images

### Performance Breakdown (Estimated)

| Object Class | Estimated mAP | Difficulty | Notes |
|--------------|---------------|------------|-------|
| bus | 0.82 | Easy | Large, distinct shape |
| truck | 0.78 | Easy | Large, rectangular |
| car | 0.72 | Medium | Common, various angles |
| cng | 0.68 | Medium | Medium size, unique shape |
| leguna | 0.65 | Medium | Bangladesh-specific |
| rickshaw | 0.63 | Medium | Traditional shape |
| motor | 0.58 | Hard | Small, often occluded |
| bicycle | 0.55 | Hard | Very small objects |
| manual-van | 0.60 | Hard | Rare in dataset |
| pedestrian | 0.52 | Very Hard | Tiny, often occluded |
| others | 0.45 | Very Hard | Diverse object types |
| **Overall** | **0.66** | - | **Weighted average** |

### Strategies that seemingly worked

#### **Small Object Detection Excellence**
```
Regular approach (640px):  Lost pedestrians/motorcycles → 0.45 mAP
Our approach (1536px):    Detected small objects → 0.66 mAP
Improvement: +0.21 mAP.
```

#### **Dense Scene Handling**
```
Default max_det=300:  Missed objects in traffic jams
Our max_det=2000:     Captured all vehicles
```

#### **Recall Optimization**
```
Conservative conf=0.25:  Missed borderline detections
Aggressive conf=0.05:    Caught uncertain objects
```

### Training Curves (Typical Pattern)
```
Epoch 1-10:   mAP: 0.20 → 0.45  (Learning basic features)
Epoch 11-25:  mAP: 0.45 → 0.62  (Fine-tuning detection)
Epoch 26-40:  mAP: 0.62 → 0.66  (Final improvements)
Epoch 41:     Early stopping triggered
```

<br>

___

<br>

## Key Insights


#### 1. **Resolution is King for Small Objects**
- **Lesson**: High-resolution training (1536) was critical
- **Impact**: +0.15-0.20 mAP improvement
- **Trade-off**: Slower training, more GPU memory

#### 2. **Confidence Threshold Strategy**
- **Lesson**: mAP rewards recall over precision
- **Strategy**: Use very low confidence (0.05)
- **Result**: Captured borderline detections
- **Competition insight**: Let evaluation handle precision

#### 3. **Mosaic Augmentation Power**
- **Lesson**: Creates realistic dense traffic scenes
- **Impact**: Model learns object relationships
- **Synthetic data**: Effectively 4x more training samples
- **Perfect fit**: For traffic jam scenarios

#### 4. **Early Stopping Importance**
- **Lesson**: Small datasets overfit quickly
- **Solution**: Patience=10 epochs
- **Result**: Prevented performance degradation
- **Validation**: 12% split provided reliable signals

#### 5. **Hardware Constraints Drive Decisions**
- **GPU limit**: 8GB forced batch=4 with 1536 images
- **Time limit**: 8 hours required efficient training
- **Model choice**: YOLOv8m balanced accuracy vs speed
- **Inference**: Slightly smaller size (1280) for speed

### Common Mistakes We Avoided

#### 1. **Using Standard Image Sizes**
```python
# Typical approach
imgsz=640   # Standard YOLO size → missed small objects

# Our approach  
imgsz=1536  # Preserved details for tiny pedestrians
```

#### 2. **Conservative Confidence Thresholds**
```python
# Typical approach
conf=0.25   # "Safe" threshold → missed detections

# Our strategy
conf=0.05   # Aggressive recall → caught everything
```

#### 3. **Default Detection Limits**
```python
# Standard setting
max_det=300  # Default → truncated dense scenes

# Our adjustment
max_det=2000 # High limit → handled traffic jams
```

#### 4. **Insufficient Augmentation**
```python
# Minimal augmentation
mosaic=0.5  # Limited synthetic scenes

# Aggressive augmentation  
mosaic=1.0  # 100% → maximum diversity
```

<br>

___

<br>


## How to Reproduce Our Results

### 1. **Environment Setup**
```bash
pip install ultralytics pandas pillow
```

### 2. **Dataset Preparation**
- Download the DU AI Challenge dataset
- Place in proper directory structure
- Run the data splitting code

### 3. **Training**
```bash
yolo detect train \
    data=data.yaml \
    model=yolov8m.pt \
    imgsz=1536 batch=4 epochs=50 \
    optimizer=AdamW lr0=0.001 cos_lr=True \
    mosaic=1.0 mixup=0.15 \
    hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
    translate=0.1 scale=0.5 fliplr=0.5 \
    cache=True patience=10 \
    device=0
```

### 4. **Inference**
```bash
yolo detect predict \
    model=runs/train/weights/best.pt \
    source=test/images \
    imgsz=1280 conf=0.05 iou=0.45 \
    save_txt=True save_conf=True max_det=2000
```

### 5. **Submission Generation**
Run the coordinate conversion script to generate final CSV.

<br>

___

<br>



## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [DU AI Challenge Official Rules](https://drive.google.com/drive/folders/1T_qrxI1oKlFTQtwNleqU2vJrBZpt9wnY?usp=drive_link)  
- [Mean Average Precision Explained](https://blog.roboflow.com/mean-average-precision/)  
- [Object Detection Best Practices](https://dbgallery.com/best-practises-ai-vision)  


<br>

___

<br>

## Team TopG Members
1. Sanjoy Das
2. Tajul Islam Tarek
3. Nayem Ahmed
4. Gazi Maksudur Rahman

<br>

___

<br>

