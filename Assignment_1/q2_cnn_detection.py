"""
Question 2.2: Load a standard CNN and apply to intersection screenshot
Uses YOLOv8 for object detection on a traffic intersection image.

Instructions:
1. Get an intersection screenshot from Florida511 (https://fl511.com)
2. Save it as 'intersection.jpg' in this folder
3. Run this script
"""

import sys
import os
from pathlib import Path

# Check for image
script_dir = Path(__file__).parent
possible_images = list(script_dir.glob("intersection*.*"))
if not possible_images:
    print("=" * 70)
    print("SETUP INSTRUCTIONS")
    print("=" * 70)
    print("""
1. Go to https://fl511.com
2. Click on a traffic camera at an intersection
3. Take a screenshot of the camera view
4. Save it as 'intersection.jpg' (or .png) in:
   """ + str(script_dir))
    print("\nThen re-run this script.")
    print("=" * 70)
    sys.exit(0)

image_path = str(possible_images[0])
print(f"Using image: {image_path}")

# Install ultralytics if needed
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics (YOLOv8)...")
    os.system(f"{sys.executable} -m pip install ultralytics")
    from ultralytics import YOLO

import cv2
import numpy as np

print("\n" + "=" * 70)
print("QUESTION 2.2: YOLO OBJECT DETECTION ON INTERSECTION")
print("=" * 70)

# Load pretrained YOLOv8 model
print("\nLoading YOLOv8 pretrained model...")
model = YOLO("yolov8n.pt")  # nano model, fast and sufficient

# Run detection
print("Running detection...")
results = model(image_path, conf=0.25)

# Process results
result = results[0]
detections = result.boxes

print(f"\n--- Detection Results ---")
print(f"Total objects detected: {len(detections)}")

# Count by class
class_counts = {}
class_confidences = {}
for box in detections:
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]
    conf = float(box.conf[0])

    if cls_name not in class_counts:
        class_counts[cls_name] = 0
        class_confidences[cls_name] = []
    class_counts[cls_name] += 1
    class_confidences[cls_name].append(conf)

print(f"\nDetections by class:")
print(f"{'Class':<20} {'Count':<8} {'Avg Confidence':<15}")
print("-" * 43)
for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    avg_conf = np.mean(class_confidences[cls_name])
    print(f"{cls_name:<20} {count:<8} {avg_conf:.3f}")

# Save annotated image
output_path = str(script_dir / "intersection_detected.jpg")
annotated = result.plot()
cv2.imwrite(output_path, annotated)
print(f"\nAnnotated image saved to: {output_path}")

# Print all individual detections
print(f"\n--- Individual Detections ---")
print(f"{'#':<4} {'Class':<15} {'Confidence':<12} {'Bounding Box (x1,y1,x2,y2)'}")
print("-" * 65)
for i, box in enumerate(detections):
    cls_name = model.names[int(box.cls[0])]
    conf = float(box.conf[0])
    coords = box.xyxy[0].cpu().numpy()
    print(f"{i+1:<4} {cls_name:<15} {conf:<12.3f} ({coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f})")

# ============================================================
# ANALYSIS AND DISCUSSION
# ============================================================
print("\n" + "=" * 70)
print("Q2.2 DISCUSSION: TRANSPORTATION PERSPECTIVE ANALYSIS")
print("=" * 70)
print("""
ACCURACY EVALUATION (by visual inspection):
- Evaluate each detection: Is the bounding box around the correct object?
- Are there missed objects (false negatives)?
- Are there incorrect detections (false positives)?
- Note the confidence scores -- lower confidence often means uncertain detections.

LIMITATIONS OF STANDARD CNN FOR TRANSPORTATION:

1. LIMITED GRANULARITY IN VEHICLE CLASSIFICATION:
   - YOLO classifies all cars as "car", all trucks as "truck"
   - Cannot distinguish: sedans vs SUVs, electric vs gas vehicles,
     emergency vehicles vs civilian, ride-share vs personal vehicles
   - No powertrain type detection (EV, hybrid, ICE)

2. PEDESTRIAN VULNERABILITY NOT CAPTURED:
   - All pedestrians classified the same -- no distinction between:
     * Children vs adults vs elderly
     * Wheelchair users or people with mobility aids
     * Jaywalkers vs crosswalk users
     * Distracted pedestrians (phone use)

3. MISSING TRANSPORTATION-SPECIFIC DETAILS:
   - No detection of traffic signal states (red/green/yellow)
   - No lane-level positioning or trajectory prediction
   - No speed or heading estimation
   - No detection of traffic violations
   - No weather/visibility condition assessment

4. SAFETY ANALYSIS GAPS:
   - Cannot assess near-miss events or conflict severity
   - No understanding of right-of-way or traffic rules
   - Cannot detect aggressive driving behavior
   - No temporal analysis (just single-frame detection)

DESIRED IMPROVEMENTS:
   - Fine-grained vehicle classification (make, model, type, powertrain)
   - Pedestrian attribute recognition (age group, vulnerability indicators)
   - Traffic signal and sign detection/recognition
   - Multi-frame tracking for trajectory and speed estimation
   - Anomaly detection for safety-critical events
   - Scene-level understanding (intersection geometry, conflict zones)
""")
