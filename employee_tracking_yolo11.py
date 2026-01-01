"""
Employee Tracking System with YOLO11m
Features:
- Person detection using YOLO11m (most accurate YOLO model)
- Sitting/Standing detection using pose estimation
- Mobile phone usage detection
- Standalone script - just provide video path
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
import urllib.request
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics package not found!")
    print("Please install: pip install ultralytics")
    exit(1)

class EnhancedEmployeeTracker:
    def __init__(self):
        # Configuration
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.output_dir = "output_frames"
        self.model_dir = "yolo_model"
        
        # Model variables
        self.yolo_model = None
        self.pose_model = None  # YOLO11 pose model
        
        # Pose keypoint indices (COCO format)
        self.POSE_PAIRS = [
            [0, 1], [1, 2], [2, 3], [3, 4],  # Head
            [1, 5], [5, 6], [6, 7],  # Left arm
            [1, 8], [8, 9], [9, 10],  # Right arm
            [1, 11], [11, 12], [12, 13],  # Left leg
            [1, 14], [14, 15], [15, 16]  # Right leg
        ]
        
        # COCO Keypoints
        self.KEYPOINTS = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
            "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
            "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
            "LEye", "REar", "LEar"
        ]
        
        # YOLO11 class names (COCO dataset)
        self.CLASS_PERSON = 0
        self.CLASS_CELL_PHONE = 67
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Create logs directory
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
        # Initialize log file
        self.log_file_path = os.path.join("logs", f"yolo11_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(self.log_file_path, "w") as f:
            f.write("# Enhanced Employee Tracking System (YOLO11m) Log\n\n")
    
    def log_event(self, message):
        """Log an event with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {message}"
        print(log_message)
        with open(self.log_file_path, "a") as log_file:
            log_file.write(f"{log_message}\n")
    
    def setup_yolo11_model(self):
        """Setup YOLO11m model"""
        self.log_event("Setting up YOLO11m model...")
        
        try:
            # YOLO11m will be automatically downloaded by Ultralytics on first use
            # This is much simpler than the old YOLOv4 approach
            self.yolo_model = YOLO('yolo11m.pt')  # Medium size for balance of speed and accuracy
            
            # Set device (CPU or GPU)
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo_model.to(device)
            
            self.log_event(f"YOLO11m model loaded successfully on {device.upper()}")
            if device == 'cuda':
                self.log_event("GPU acceleration enabled!")
            else:
                self.log_event("Running on CPU (consider using GPU for faster processing)")
            
            return True
        except Exception as e:
            self.log_event(f"Error loading YOLO11m model: {e}")
            return False
    
    def setup_pose_model(self):
        """Setup YOLO11-pose model for pose estimation"""
        self.log_event("Setting up YOLO11-pose model...")
        
        try:
            # Use YOLO11's built-in pose estimation model
            # This eliminates the need for separate OpenPose download
            self.pose_model = YOLO('yolo11m-pose.pt')
            
            # Set device (CPU or GPU)
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.pose_model.to(device)
            
            self.log_event(f"YOLO11-pose model loaded successfully on {device.upper()}")
            return True
        except Exception as e:
            self.log_event(f"Error loading YOLO11-pose model: {e}")
            return False
    
    def detect_persons_yolo11(self, frame):
        """Detect persons and phones using YOLO11m"""
        try:
            # Run YOLO11 inference
            results = self.yolo_model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]
            
            person_boxes = []
            phone_boxes = []
            
            # Process detections
            if results.boxes is not None:
                for box in results.boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    # Filter by class
                    if class_id == self.CLASS_PERSON:
                        person_boxes.append([x, y, w, h, confidence])
                    elif class_id == self.CLASS_CELL_PHONE:
                        phone_boxes.append([x, y, w, h, confidence])
            
            return person_boxes, phone_boxes
            
        except Exception as e:
            self.log_event(f"Error during YOLO11 detection: {e}")
            return [], []
    
    def detect_pose_keypoints(self, frame, person_box):
        """Detect pose keypoints for a person using YOLO11-pose"""
        x, y, w, h, conf = person_box
        
        # Crop person region with some padding
        padding = 20
        y1 = max(0, y - padding)
        y2 = min(frame.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(frame.shape[1], x + w + padding)
        
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return None
        
        try:
            # Run YOLO11-pose on the person ROI
            results = self.pose_model(person_roi, conf=0.3, verbose=False)[0]
            
            # Check if any keypoints were detected
            if results.keypoints is None or len(results.keypoints) == 0:
                return None
            
            # Get the keypoints (YOLO11 uses COCO format with 17 keypoints)
            # YOLO11 keypoints format: [x, y, confidence] for each keypoint
            kpts = results.keypoints.data[0].cpu().numpy()  # Get first person's keypoints
            
            keypoints = []
            for i in range(len(kpts)):
                x_key = int(kpts[i][0]) + x1
                y_key = int(kpts[i][1]) + y1
                kpt_conf = float(kpts[i][2])
                
                if kpt_conf > 0.3:  # Threshold for keypoint confidence
                    keypoints.append((x_key, y_key, kpt_conf))
                else:
                    keypoints.append(None)
            
            # YOLO11-pose has 17 keypoints, we need to add one more for compatibility
            # Insert neck at index 1 (between nose and shoulders)
            if len(keypoints) >= 6 and keypoints[5] and keypoints[6]:
                # Calculate neck position from shoulders
                neck_x = int((keypoints[5][0] + keypoints[6][0]) / 2)
                neck_y = int((keypoints[5][1] + keypoints[6][1]) / 2)
                neck_conf = (keypoints[5][2] + keypoints[6][2]) / 2
                keypoints.insert(1, (neck_x, neck_y, neck_conf))
            else:
                keypoints.insert(1, None)
            
            return keypoints
            
        except Exception as e:
            # If pose detection fails, return None
            return None
    
    def determine_sitting_standing(self, keypoints):
        """Determine if person is sitting or standing based on pose keypoints"""
        if not keypoints or len(keypoints) < 18:
            return "Unknown", 0.0
        
        # Key indices: 1=Neck, 8=RHip, 11=LHip, 9=RKnee, 12=LKnee, 10=RAnkle, 13=LAnkle
        neck = keypoints[1]
        r_hip = keypoints[8]
        l_hip = keypoints[11]
        r_knee = keypoints[9]
        l_knee = keypoints[12]
        r_ankle = keypoints[10]
        l_ankle = keypoints[13]
        
        # Check if we have the necessary keypoints
        if not all([neck, r_hip, l_hip]):
            return "Unknown", 0.0
        
        # Calculate average hip position
        avg_hip_y = (r_hip[1] + l_hip[1]) / 2
        neck_y = neck[1]
        
        # Calculate torso length (neck to hip)
        torso_length = avg_hip_y - neck_y
        
        if torso_length <= 0:
            return "Unknown", 0.0
        
        # Calculate confidence based on keypoint visibility
        visible_keypoints = sum(1 for kp in [neck, r_hip, l_hip, r_knee, l_knee] if kp is not None)
        confidence = visible_keypoints / 5.0
        
        # Check knee positions if available
        has_knees = r_knee is not None or l_knee is not None
        has_ankles = r_ankle is not None or l_ankle is not None
        
        if has_knees:
            knee_positions = []
            if r_knee:
                knee_positions.append(r_knee[1])
            if l_knee:
                knee_positions.append(l_knee[1])
            
            avg_knee_y = sum(knee_positions) / len(knee_positions)
            
            # Calculate knee-to-hip ratio
            knee_hip_distance = avg_knee_y - avg_hip_y
            ratio = knee_hip_distance / torso_length
            
            # Enhanced logic with ankles for better accuracy
            if has_ankles:
                ankle_positions = []
                if r_ankle:
                    ankle_positions.append(r_ankle[1])
                if l_ankle:
                    ankle_positions.append(l_ankle[1])
                
                if ankle_positions:
                    avg_ankle_y = sum(ankle_positions) / len(ankle_positions)
                    ankle_knee_distance = avg_ankle_y - avg_knee_y
                    
                    # If ankles are very close to knees, likely sitting
                    if ankle_knee_distance < torso_length * 0.3:
                        return "Sitting", min(confidence * 1.2, 1.0)
            
            # If knees are close to hips (small distance), person is likely sitting
            # If knees are far from hips (large distance), person is standing
            if ratio < 0.4:  # More strict threshold
                return "Sitting", confidence
            elif ratio > 0.9:  # More strict threshold
                return "Standing", confidence
            elif ratio < 0.6:
                return "Sitting", confidence * 0.8
            else:
                return "Standing", confidence * 0.8
        else:
            # Fallback: use torso angle if knees not detected
            return "Unknown (No leg detection)", 0.3
    
    def check_phone_usage(self, person_box, phone_boxes, keypoints=None):
        """Check if person is using a mobile phone with enhanced accuracy"""
        if not phone_boxes:
            return False, 0.0
        
        px, py, pw, ph, _ = person_box
        person_center = (px + pw/2, py + ph/2)
        person_radius = max(pw, ph) / 2
        
        # Check if we have hand/wrist keypoints for more accurate detection
        hand_regions = []
        if keypoints:
            # Get wrist positions (4=RWrist, 7=LWrist)
            r_wrist = keypoints[4] if len(keypoints) > 4 and keypoints[4] else None
            l_wrist = keypoints[7] if len(keypoints) > 7 and keypoints[7] else None
            
            if r_wrist:
                hand_regions.append((r_wrist[0], r_wrist[1]))
            if l_wrist:
                hand_regions.append((l_wrist[0], l_wrist[1]))
        
        best_match_distance = float('inf')
        
        for phone_box in phone_boxes:
            phone_x, phone_y, phone_w, phone_h, phone_conf = phone_box
            phone_center = (phone_x + phone_w/2, phone_y + phone_h/2)
            
            # Check if phone is near hands first (more accurate)
            if hand_regions:
                for hand_pos in hand_regions:
                    hand_distance = np.sqrt((hand_pos[0] - phone_center[0])**2 + 
                                          (hand_pos[1] - phone_center[1])**2)
                    # If phone is near hand (within 150 pixels), high confidence
                    if hand_distance < 150:
                        return True, 0.9
                    best_match_distance = min(best_match_distance, hand_distance)
            
            # Calculate distance between person center and phone
            distance = np.sqrt((person_center[0] - phone_center[0])**2 + 
                             (person_center[1] - phone_center[1])**2)
            
            # If phone is within person's bounding box
            if distance < person_radius * 1.3:
                # Check if phone is in upper body region (more likely to be in use)
                upper_body_y = py + ph * 0.3  # Upper 30% of body
                if phone_y < upper_body_y:
                    return True, 0.8
                else:
                    return True, 0.6
        
        return False, 0.0
    
    def draw_skeleton(self, frame, keypoints):
        """Draw skeleton connections between keypoints"""
        if not keypoints:
            return
        
        # Define skeleton connections (COCO format)
        skeleton = [
            [1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
            [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
            [0, 14], [14, 16], [0, 15], [15, 17]
        ]
        
        for pair in skeleton:
            if len(keypoints) > max(pair):
                kp1 = keypoints[pair[0]]
                kp2 = keypoints[pair[1]]
                
                if kp1 and kp2:
                    cv2.line(frame, (kp1[0], kp1[1]), (kp2[0], kp2[1]), 
                            (0, 255, 255), 2, cv2.LINE_AA)
    
    def process_video(self, video_path):
        """Process video and track employees"""
        self.log_event(f"Opening video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.log_event(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.log_event(f"Video FPS: {fps}, Total Frames: {total_frames}")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(self.output_dir, f"yolo11_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        out = None
        
        frame_count = 0
        last_save_time = time.time()
        processing_times = []
        
        # Statistics
        total_persons_detected = 0
        total_phone_usage = 0
        total_working = 0
        total_standing = 0
        total_idle = 0
        posture_stats = {"Sitting": 0, "Standing": 0, "Unknown": 0}
        activity_stats = {"Working": 0, "Using Phone": 0, "Away from Desk": 0, "Idle": 0}
        
        try:
            while True:
                frame_start = time.time()
                ret, frame = cap.read()
                
                if not ret:
                    self.log_event("End of video reached")
                    break
                
                frame_count += 1
                
                # Resize frame for optimal processing
                original_frame = frame.copy()
                max_width = 1280
                if frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                # Initialize output video writer with first frame dimensions
                if out is None:
                    height, width = frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Detect persons and phones using YOLO11
                person_boxes, phone_boxes = self.detect_persons_yolo11(frame)
                
                total_persons_detected += len(person_boxes)
                
                # Process each detected person
                for person_box in person_boxes:
                    x, y, w, h, conf = person_box
                    
                    # Detect pose keypoints
                    keypoints = self.detect_pose_keypoints(frame, person_box)
                    
                    # Determine sitting/standing
                    posture = "Unknown"
                    posture_conf = 0.0
                    if keypoints:
                        posture, posture_conf = self.determine_sitting_standing(keypoints)
                        posture_stats[posture.split()[0]] = posture_stats.get(posture.split()[0], 0) + 1
                        
                        # Draw skeleton
                        self.draw_skeleton(frame, keypoints)
                        
                        # Draw keypoints
                        for i, kp in enumerate(keypoints):
                            if kp:
                                cv2.circle(frame, (kp[0], kp[1]), 4, (0, 255, 255), -1)
                                cv2.circle(frame, (kp[0], kp[1]), 5, (0, 0, 0), 1)
                    
                    # Check phone usage with enhanced accuracy
                    using_phone, phone_conf = self.check_phone_usage(person_box, phone_boxes, keypoints)
                    
                    if using_phone:
                        total_phone_usage += 1
                    
                    # Determine activity status based on posture and phone usage
                    # Logic: Only sitting employees can be "working"
                    if using_phone:
                        activity = f"Using Phone ({phone_conf:.1%})"
                        box_color = (0, 0, 255)  # Red
                        activity_stats["Using Phone"] += 1
                    elif posture == "Sitting":
                        activity = "Working (Sitting)"
                        box_color = (0, 255, 0)  # Green
                        activity_stats["Working"] += 1
                        total_working += 1
                    elif posture == "Standing":
                        activity = "Away from Desk (Standing)"
                        box_color = (0, 165, 255)  # Orange
                        activity_stats["Away from Desk"] += 1
                        total_standing += 1
                    else:
                        activity = "Idle/Moving"
                        box_color = (0, 255, 255)  # Yellow
                        activity_stats["Idle"] += 1
                        total_idle += 1
                    
                    # Draw person bounding box with thickness based on confidence
                    thickness = 2 if conf > 0.7 else 1
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, thickness)
                    
                    # Create info box background
                    label_y = y - 50 if y > 60 else y + h + 10
                    cv2.rectangle(frame, (x, label_y - 45), (x + w, label_y), (0, 0, 0), -1)
                    cv2.rectangle(frame, (x, label_y - 45), (x + w, label_y), box_color, 2)
                    
                    # Add labels with better formatting
                    cv2.putText(frame, f"Posture: {posture}", (x + 5, label_y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"Activity: {activity}", (x + 5, label_y - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"Confidence: {conf:.1%}", (x + 5, label_y - 0),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                
                # Draw phone detections
                for phone_box in phone_boxes:
                    px, py, pw, ph, pconf = phone_box
                    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 0, 255), 2)
                    cv2.putText(frame, f"Phone {pconf:.1%}", (px, py-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
                
                # Add enhanced frame info with dark background
                info_height = 120
                cv2.rectangle(frame, (0, 0), (400, info_height), (0, 0, 0), -1)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"YOLO11m Employee Tracker", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)", 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Time: {timestamp}", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Persons: {len(person_boxes)} | Phones: {len(phone_boxes)}", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # FPS counter
                frame_time = time.time() - frame_start
                processing_times.append(frame_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)
                avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
                cv2.putText(frame, f"Processing: {avg_fps:.1f} FPS", (10, 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Write frame to output video
                out.write(frame)
                
                # Display frame
                cv2.imshow('YOLO11m Employee Tracking', frame)
                
                # Save frame periodically
                if time.time() - last_save_time > 10:
                    frame_filename = os.path.join(self.output_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    last_save_time = time.time()
                
                # Log progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.log_event(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - Avg FPS: {avg_fps:.1f}")
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.log_event("Processing interrupted by user")
                    break
        
        except Exception as e:
            self.log_event(f"Error during video processing: {e}")
            import traceback
            self.log_event(traceback.format_exc())
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Log final statistics
            self.log_event("="*60)
            self.log_event("PROCESSING SUMMARY")
            self.log_event("="*60)
            self.log_event(f"Output saved to: {output_path}")
            self.log_event(f"Total frames processed: {frame_count}")
            self.log_event(f"Total persons detected: {total_persons_detected}")
            self.log_event("")
            self.log_event("ACTIVITY BREAKDOWN:")
            self.log_event(f"  - Working (Sitting): {total_working}")
            self.log_event(f"  - Away from Desk (Standing): {total_standing}")
            self.log_event(f"  - Phone usage instances: {total_phone_usage}")
            self.log_event(f"  - Idle/Moving: {total_idle}")
            self.log_event("")
            self.log_event(f"Posture statistics: {posture_stats}")
            self.log_event(f"Activity statistics: {activity_stats}")
            
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            self.log_event(f"Average processing time per frame: {avg_processing_time:.3f}s")
            self.log_event(f"Average FPS: {1.0/avg_processing_time:.1f}" if avg_processing_time > 0 else "N/A")

def main():
    """Main function to run the enhanced employee tracker"""
    parser = argparse.ArgumentParser(description='Enhanced Employee Tracking System with YOLO11m')
    parser.add_argument('video_path', type=str, help='Path to the video file or 0 for webcam')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold for detection (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold for NMS (default: 0.45)')
    
    args = parser.parse_args()
    
    # Create tracker instance
    tracker = EnhancedEmployeeTracker()
    tracker.confidence_threshold = args.confidence
    tracker.iou_threshold = args.iou
    
    # Setup models
    print("\n" + "="*60)
    print("ENHANCED EMPLOYEE TRACKING SYSTEM - YOLO11m")
    print("="*60 + "\n")
    
    print("Setting up models...\n")
    
    if not tracker.setup_yolo11_model():
        print("Failed to setup YOLO11m model. Exiting.")
        return
    
    if not tracker.setup_pose_model():
        print("Failed to setup pose estimation model. Exiting.")
        return
    
    print("\nModels loaded successfully!")
    print("="*60 + "\n")
    
    # Process video
    video_path = args.video_path
    if video_path == '0':
        video_path = 0
    
    tracker.process_video(video_path)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nCheck the '{tracker.output_dir}' folder for output video and frames")
    print(f"Check the 'logs' folder for detailed logs\n")

if __name__ == "__main__":
    main()
