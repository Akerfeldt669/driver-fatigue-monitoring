"""
Real-Time Driver Drowsiness Detection
Main application with camera feed and live detection
"""
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import time
from collections import deque
import os
import sys
from pathlib import Path
import winsound  # Windows sound library (built-in)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================
# CONFIGURATION
# ============================================

# Model paths
EYE_MODEL_PATH = r"C:\Users\ADMIN\Desktop\driver-fatigue-monitoring\models\saved\eye_classifier.h5"
YAWN_MODEL_PATH = r"C:\Users\ADMIN\Desktop\driver-fatigue-monitoring\models\saved\yawn_detector.h5"

# Detection thresholds
EAR_THRESHOLD = 0.21        # Eye Aspect Ratio threshold (lower = more sensitive)
DROWSY_FRAMES = 15          # Consecutive frames to trigger drowsy alert
YAWN_FRAMES = 12            # Consecutive frames to trigger yawn alert
CONFIDENCE_THRESHOLD = 0.65 # ML model confidence threshold for yawn

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def preprocess_mouth(mouth_roi):
    """Preprocess mouth ROI for model input"""
    try:
        # Resize to 48x48
        mouth_roi = cv2.resize(mouth_roi, (48, 48))
        
        # Ensure RGB
        if len(mouth_roi.shape) == 2:
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_GRAY2RGB)
        else:
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2RGB)
        
        # Normalize
        mouth_roi = mouth_roi.astype(np.float32) / 255.0
        
        # Add batch dimension
        mouth_roi = np.expand_dims(mouth_roi, axis=0)
        
        return mouth_roi
    except:
        return None


# ============================================
# MAIN DETECTOR CLASS
# ============================================

class DrowsinessDetector:
    def __init__(self):
        """Initialize the drowsiness detector"""
        print("="*70)
        print("INITIALIZING DROWSINESS DETECTOR")
        print("="*70)
        print()
        
        # Load models
        print("Loading models...")
        self.load_models()
        
        # Initialize Mediapipe Face Mesh
        print("Loading Mediapipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for EAR calculation
        # Left eye: [p1, p2, p3, p4, p5, p6]
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
        # Mouth landmarks for yawn detection
        self.MOUTH_INDICES = [61, 291, 0, 17, 269, 405, 39, 181]
        
        # State tracking
        self.drowsy_counter = 0
        self.yawn_counter = 0
        self.total_blinks = 0
        self.fps_history = deque(maxlen=30)
        
        # Statistics
        self.stats = {
            'drowsy_events': 0,
            'yawn_events': 0,
            'total_frames': 0
        }
        
        print("✓ Initialization complete!")
        print()
    
    def load_models(self):
        """Load trained models"""
        # Load yawn model
        if os.path.exists(YAWN_MODEL_PATH):
            self.yawn_model = keras.models.load_model(YAWN_MODEL_PATH)
            print(f"✓ Yawn model loaded: {os.path.abspath(YAWN_MODEL_PATH)}")
        else:
            self.yawn_model = None
            print(f"⚠️  Yawn model not found: {YAWN_MODEL_PATH}")
            print("   Yawn detection will be disabled.")
    
    def extract_roi(self, frame, landmarks, indices, padding=20):
        """Extract Region of Interest (ROI) from frame"""
        h, w = frame.shape[:2]
        
        # Get landmark coordinates
        points = []
        for idx in indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            points.append([x, y])
        
        points = np.array(points)
        
        # Get bounding box
        x_min = max(0, np.min(points[:, 0]) - padding)
        y_min = max(0, np.min(points[:, 1]) - padding)
        x_max = min(w, np.max(points[:, 0]) + padding)
        y_max = min(h, np.max(points[:, 1]) + padding)
        
        # Extract ROI
        roi = frame[y_min:y_max, x_min:x_max]
        
        return roi, (x_min, y_min, x_max, y_max)
    
    def calculate_ear(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_points: Array of 6 eye landmark points
        
        Returns:
            EAR value (float)
        """
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # EAR calculation
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear
    
    def detect_drowsiness(self, frame):
        """
        Main detection function using HYBRID approach:
        - EAR (geometric) for eye detection - more reliable
        - ML model for yawn detection - more accurate
        
        Returns: annotated_frame, status, status_color, info
        """
        start_time = time.time()
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize status
        status = "ACTIVE"
        status_color = COLOR_GREEN
        info = {}
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # ============================================
            # EYE DETECTION USING EAR (GEOMETRIC METHOD)
            # ============================================
            
            # Get left eye landmarks
            left_eye_points = np.array([
                [face_landmarks[idx].x * w, face_landmarks[idx].y * h] 
                for idx in self.LEFT_EYE_INDICES
            ])
            
            # Get right eye landmarks
            right_eye_points = np.array([
                [face_landmarks[idx].x * w, face_landmarks[idx].y * h] 
                for idx in self.RIGHT_EYE_INDICES
            ])
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye_points)
            right_ear = self.calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Determine if eyes are closed based on EAR
            eyes_closed = avg_ear < EAR_THRESHOLD
            
            # Calculate bounding boxes
            left_bbox = (
                int(np.min(left_eye_points[:, 0]) - 10),
                int(np.min(left_eye_points[:, 1]) - 10),
                int(np.max(left_eye_points[:, 0]) + 10),
                int(np.max(left_eye_points[:, 1]) + 10)
            )
            
            right_bbox = (
                int(np.min(right_eye_points[:, 0]) - 10),
                int(np.min(right_eye_points[:, 1]) - 10),
                int(np.max(right_eye_points[:, 0]) + 10),
                int(np.max(right_eye_points[:, 1]) + 10)
            )
            
            # Color based on eye state
            eye_color = COLOR_RED if eyes_closed else COLOR_GREEN
            
            # Draw bounding boxes around eyes
            cv2.rectangle(frame, (left_bbox[0], left_bbox[1]), (left_bbox[2], left_bbox[3]), eye_color, 2)
            cv2.rectangle(frame, (right_bbox[0], right_bbox[1]), (right_bbox[2], right_bbox[3]), eye_color, 2)
            
            # Draw EAR value on left eye
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (left_bbox[0], left_bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 2)
            
            # Update drowsy counter based on EAR
            if eyes_closed:
                self.drowsy_counter += 1
            else:
                if self.drowsy_counter > 3:  # Only count as blink if eyes were closed for few frames
                    self.total_blinks += 1
                self.drowsy_counter = 0
            
            # Determine status based on drowsy counter
            if self.drowsy_counter >= DROWSY_FRAMES:
                status = "DROWSY!"
                status_color = COLOR_RED
                if self.drowsy_counter == DROWSY_FRAMES:  # Count only once
                    self.stats['drowsy_events'] += 1
            elif self.drowsy_counter > DROWSY_FRAMES // 2:
                status = "WARNING"
                status_color = COLOR_ORANGE
            
            # Store eye info
            info['left_ear'] = float(left_ear)
            info['right_ear'] = float(right_ear)
            info['avg_ear'] = float(avg_ear)
            info['eyes_closed'] = eyes_closed
            info['ear_threshold'] = EAR_THRESHOLD
            
            # ============================================
            # YAWN DETECTION USING ML MODEL
            # ============================================
            
            if self.yawn_model is not None:
                mouth_roi, mouth_bbox = self.extract_roi(frame, face_landmarks, self.MOUTH_INDICES, padding=15)
                
                if mouth_roi.size > 0:
                    # Preprocess
                    mouth_processed = preprocess_mouth(mouth_roi)
                    
                    if mouth_processed is not None:
                        # Predict
                        yawn_pred = self.yawn_model.predict(mouth_processed, verbose=0)[0]
                        
                        # Get prediction (0=no_yawn/normal, 1=yawn/yawning)
                        is_yawning = yawn_pred[1] > CONFIDENCE_THRESHOLD
                        
                        # Draw mouth bounding box
                        mouth_color = COLOR_YELLOW if is_yawning else COLOR_GREEN
                        cv2.rectangle(frame, (mouth_bbox[0], mouth_bbox[1]), (mouth_bbox[2], mouth_bbox[3]), mouth_color, 2)
                        
                        # Draw yawn confidence
                        cv2.putText(frame, f"Yawn: {yawn_pred[1]:.2f}", (mouth_bbox[0], mouth_bbox[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mouth_color, 2)
                        
                        # Update yawn counter
                        if is_yawning:
                            self.yawn_counter += 1
                        else:
                            self.yawn_counter = 0
                        
                        # Check if yawning
                        if self.yawn_counter >= YAWN_FRAMES:
                            if status == "ACTIVE":  # Only override if not drowsy
                                status = "YAWNING"
                                status_color = COLOR_YELLOW
                            if self.yawn_counter == YAWN_FRAMES:  # Count only once
                                self.stats['yawn_events'] += 1
                        
                        # Store yawn info
                        info['yawn_conf'] = float(yawn_pred[1])
                        info['is_yawning'] = is_yawning
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = 1.0 / (elapsed + 1e-6)
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        
        # Update stats
        self.stats['total_frames'] += 1
        info['fps'] = avg_fps
        info['status'] = status
        info['status_color'] = status_color
        
        return frame, status, status_color, info
    
    def draw_ui(self, frame, status, status_color, info):
        """Draw UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for status panel
        overlay = frame.copy()
        
        # Top status bar
        cv2.rectangle(overlay, (0, 0), (w, 100), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status text (increased thickness for bold effect)
        cv2.putText(frame, f"STATUS: {status}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 4)
        
        # FPS
        fps_text = f"FPS: {info.get('fps', 0):.1f}"
        cv2.putText(frame, fps_text, (w - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
        
        # Stats panel (bottom left)
        y_offset = h - 150
        cv2.rectangle(frame, (0, y_offset), (300, h), COLOR_BLACK, -1)
        
        cv2.putText(frame, "STATISTICS", (10, y_offset + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        
        cv2.putText(frame, f"Frames: {self.stats['total_frames']}", (10, y_offset + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        
        cv2.putText(frame, f"Drowsy Events: {self.stats['drowsy_events']}", (10, y_offset + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        
        cv2.putText(frame, f"Yawn Events: {self.stats['yawn_events']}", (10, y_offset + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, y_offset + 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        
        # Detection info (bottom right)
        if 'avg_ear' in info:
            x_offset = w - 280
            cv2.rectangle(frame, (x_offset, y_offset), (w, h), COLOR_BLACK, -1)
            
            cv2.putText(frame, "DETECTION", (x_offset + 10, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
            
            # Eye status with EAR value
            eye_status = "CLOSED" if info['eyes_closed'] else "OPEN"
            eye_color = COLOR_RED if info['eyes_closed'] else COLOR_GREEN
            cv2.putText(frame, f"Eyes: {eye_status}", (x_offset + 10, y_offset + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 2)
            
            # Show EAR value
            ear_text = f"EAR: {info['avg_ear']:.3f}"
            ear_display_color = COLOR_RED if info['avg_ear'] < EAR_THRESHOLD else COLOR_GREEN
            cv2.putText(frame, ear_text, (x_offset + 10, y_offset + 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, ear_display_color, 1)
            
            if 'is_yawning' in info:
                yawn_status = "YES" if info['is_yawning'] else "NO"
                yawn_color = COLOR_YELLOW if info['is_yawning'] else COLOR_GREEN
                cv2.putText(frame, f"Yawn: {yawn_status}", (x_offset + 10, y_offset + 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, yawn_color, 2)
                
                # Show yawn confidence
                yawn_conf_text = f"Conf: {info['yawn_conf']:.2f}"
                cv2.putText(frame, yawn_conf_text, (x_offset + 10, y_offset + 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
            
            # Drowsy counter
            if self.drowsy_counter > 0:
                cv2.putText(frame, f"Drowsy: {self.drowsy_counter}/{DROWSY_FRAMES}", 
                           (x_offset + 10, y_offset + 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ORANGE, 2)
        
        # Instructions (top right)
        cv2.putText(frame, "Press 'Q' to quit", (w - 200, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        
        return frame
    
    def run(self):
        """Main run loop"""
        print("="*70)
        print("STARTING REAL-TIME DETECTION")
        print("="*70)
        print("Press 'Q' to quit")
        print()
        
        # Open camera
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("✓ Camera opened successfully")
        print()
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Detect drowsiness
                frame, status, status_color, info = self.detect_drowsiness(frame)
                
                # Draw UI
                frame = self.draw_ui(frame, status, status_color, info)
                
                # Show frame
                cv2.imshow("Driver Drowsiness Detection", frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final stats
            print()
            print("="*70)
            print("SESSION SUMMARY")
            print("="*70)
            print(f"Total Frames: {self.stats['total_frames']}")
            print(f"Drowsy Events: {self.stats['drowsy_events']}")
            print(f"Yawn Events: {self.stats['yawn_events']}")
            print(f"Total Blinks: {self.total_blinks}")
            print("="*70)


# ============================================
# MAIN
# ============================================

def main():
    """Main entry point"""
    try:
        detector = DrowsinessDetector()
        detector.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()