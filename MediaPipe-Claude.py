import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# =====================================================================
# CONFIGURATION - OPTIMIZED FOR YOUR SPECIFIC VIDEO
# =====================================================================
VIDEO_PATH = r"D:\Tremor Videos Subject #1\Day15-Before_rotated.mp4"  # Change to your video file
OUTPUT_VIDEO_PATH = "Day15-Before_annotated.mp4"
CSV_OUTPUT_PATH = "Day15-Before.csv"

# =====================================================================
# SKIN DETECTION OPTIMIZED FOR YOUR VIDEO
# =====================================================================
def detect_skin_optimized(frame):
    """Detect skin with parameters optimized for the shown video"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Optimize for beige/light brown skin against white background
    # Narrower hue range (focusing on the specific skin tone shown)
    # Lower saturation minimum to catch the less saturated parts
    # Higher value minimum to distinguish from white background
    lower_skin = np.array([5, 10, 100], dtype=np.uint8)
    upper_skin = np.array([25, 150, 255], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # More aggressive cleaning to isolate hands
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Additional processing to separate hands
    mask = cv2.medianBlur(mask, 5)
    
    return mask

def detect_hands_by_position(skin_mask, frame_width):
    """Separate left and right hands based on position in image"""
    # Find contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area to filter small noise
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    left_hand = None
    right_hand = None
    
    # Extract top 2 contours which should be hands
    hand_contours = []
    min_area = 3000  # Minimum area to consider as a hand
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            hand_contours.append(contour)
            if len(hand_contours) >= 2:  # Stop after finding 2 hands
                break
    
    # If we found exactly 2 hands
    if len(hand_contours) == 2:
        # Determine which is left and right based on x-position
        # Get centers of each contour
        centers = []
        for contour in hand_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                centers.append(cx)
            else:
                centers.append(0)
        
        # The hand with smaller x-coordinate is on the left side of the image
        if centers[0] < centers[1]:
            left_hand = hand_contours[0]
            right_hand = hand_contours[1]
        else:
            left_hand = hand_contours[1]
            right_hand = hand_contours[0]
    
    # If we only found one hand, try to determine if it's left or right
    elif len(hand_contours) == 1:
        M = cv2.moments(hand_contours[0])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            # Determine if it's on the left or right side of the frame
            if cx < frame_width / 2:
                left_hand = hand_contours[0]
            else:
                right_hand = hand_contours[0]
    
    return left_hand, right_hand

def create_hand_mask(contour, shape):
    """Create a binary mask for a single hand"""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    return mask

def estimate_hand_keypoints(contour, is_left, frame_shape):
    """Estimate hand keypoints from contour"""
    # Calculate bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate moments for center
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w//2, y + h//2
    
    # Simple keypoint estimation
    keypoints = []
    
    # Format: (x, y, point_type)
    # 0: wrist
    # 1-4: thumb points
    # 5-8: index finger points
    # 9-12: middle finger points
    # 13-16: ring finger points
    # 17-20: pinky points
    
    # Wrist (center bottom)
    wrist_y = y + h - int(h*0.1)  # Slightly above bottom of bounding box
    keypoints.append((cx, wrist_y, 0))  # Wrist
    
    # Palm center
    keypoints.append((cx, cy, 9))  # Using middle finger MCP position
    
    # Finger estimation - simplistic approach
    finger_width = w / 6
    finger_heights = [0.4, 0.3, 0.2, 0.3, 0.4]  # Height ratios for fingers
    
    # Base positions for each finger (thumb and 4 fingers)
    bases = []
    for i in range(5):
        base_x = x + int((i+1) * finger_width)
        base_y = y + int(h * 0.5)  # Middle of hand
        bases.append((base_x, base_y))
    
    # For left hand, reverse the order of fingers
    if is_left:
        bases.reverse()
    
    # Add thumb (special case)
    thumb_base_x = bases[0][0]
    thumb_base_y = bases[0][1]
    thumb_tip_x = thumb_base_x + (-1 if is_left else 1) * int(w * 0.2)
    thumb_tip_y = thumb_base_y - int(h * 0.2)
    
    keypoints.append((thumb_base_x, thumb_base_y, 1))  # CMC
    keypoints.append((thumb_base_x, thumb_base_y - int(h*0.1), 2))  # MCP
    keypoints.append((thumb_tip_x, thumb_base_y - int(h*0.15), 3))  # IP
    keypoints.append((thumb_tip_x, thumb_tip_y, 4))  # TIP
    
    # Add other 4 fingers
    point_idx = 5
    for i in range(1, 5):
        base_x, base_y = bases[i]
        
        # MCP joint (knuckle)
        keypoints.append((base_x, base_y, point_idx))
        point_idx += 1
        
        # PIP joint (middle joint)
        keypoints.append((base_x, base_y - int(h*0.2), point_idx))
        point_idx += 1
        
        # DIP joint
        keypoints.append((base_x, base_y - int(h*0.3), point_idx))
        point_idx += 1
        
        # Tip
        tip_y = base_y - int(h * finger_heights[i])
        keypoints.append((base_x, tip_y, point_idx))
        point_idx += 1
    
    # Normalize keypoints
    normalized_keypoints = []
    for x, y, idx in keypoints:
        norm_x = max(0, min(1.0, x / frame_shape[1]))  # Keep within [0, 1]
        norm_y = max(0, min(1.0, y / frame_shape[0]))
        normalized_keypoints.append((norm_x, norm_y, idx))
    
    return normalized_keypoints

def process_video():
    print(f"Processing video: {VIDEO_PATH}")
    
    # Initialize MediaPipe
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Lower threshold
            min_tracking_confidence=0.3
        )
        mp_draw = mp.solutions.drawing_utils
        print("✅ MediaPipe initialized")
    except Exception as e:
        print(f"❌ Error initializing MediaPipe: {str(e)}")
        return
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video {VIDEO_PATH}")
        return
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Check FPS value
    if fps == 0:
        print("⚠️ WARNING: FPS is 0, setting default 30 FPS")
        fps = 30
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    
    # Data storage
    data = []
    frame_count = 0
    mediapipe_detections = 0
    opencv_detections = 0
    
    print("✅ Starting video processing...")
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("✅ Video processing complete.")
            break
        
        # Create data structure for this frame
        frame_data = {"Frame": frame_count}
        
        # Initialize columns for both hands
        for hand_label in ["Left", "Right"]:
            for i in range(21):
                frame_data[f"{hand_label}_Point{i}_x"] = None
                frame_data[f"{hand_label}_Point{i}_y"] = None
        
        # Copy frame for annotations
        display_frame = frame.copy()
        
        # STAGE 1: Try MediaPipe first
        mediapipe_detected = False
        try:
            # Pre-process for MediaPipe
            # Slightly increase contrast to help with detection
            processed = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
            rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            result = hands.process(rgb_frame)
            
            # Check for hand detections
            if result.multi_hand_landmarks and result.multi_handedness:
                mediapipe_detected = True
                mediapipe_detections += 1
                
                for idx, (hand_landmarks, hand_info) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
                    # Get hand label
                    raw_label = hand_info.classification[0].label
                    hand_label = "Left" if raw_label == "Right" else "Right"
                    
                    # Get confidence
                    confidence = hand_info.classification[0].score
                    
                    # Add visual label
                    wrist_x = int(hand_landmarks.landmark[0].x * frame_width)
                    wrist_y = int(hand_landmarks.landmark[0].y * frame_height)
                    cv2.putText(display_frame, f"{hand_label} (MP)", (wrist_x, wrist_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Store landmarks
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        x = landmark.x
                        y = landmark.y
                        
                        frame_data[f"{hand_label}_Point{i}_x"] = x
                        frame_data[f"{hand_label}_Point{i}_y"] = y
                    
                    # Draw landmarks
                    color = (0, 255, 0) if hand_label == "Right" else (255, 0, 0)
                    mp_draw.draw_landmarks(
                        display_frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                
                print(f"✅ MediaPipe detected {len(result.multi_hand_landmarks)} hands in frame {frame_count}")
        
        except Exception as e:
            print(f"⚠️ MediaPipe error: {str(e)}")
        
        # STAGE 2: If MediaPipe failed, use OpenCV
        if not mediapipe_detected:
            try:
                # Detect skin using optimized parameters
                skin_mask = detect_skin_optimized(frame)
                
                # Detect left and right hands by position
                left_hand, right_hand = detect_hands_by_position(skin_mask, frame_width)
                
                hands_detected = 0
                if left_hand is not None:
                    hands_detected += 1
                if right_hand is not None:
                    hands_detected += 1
                
                if hands_detected > 0:
                    opencv_detections += 1
                    print(f"✅ OpenCV detected {hands_detected} hands in frame {frame_count}")
                    
                    # Process left hand if detected
                    if left_hand is not None:
                        cv2.drawContours(display_frame, [left_hand], 0, (255, 0, 0), 2)
                        cv2.putText(display_frame, "Left (CV)", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Estimate keypoints
                        keypoints = estimate_hand_keypoints(left_hand, True, (frame_height, frame_width))
                        
                        # Store keypoints in data
                        for x, y, idx in keypoints:
                            frame_data[f"Left_Point{idx}_x"] = x
                            frame_data[f"Left_Point{idx}_y"] = y
                            
                            # Draw keypoints
                            px, py = int(x * frame_width), int(y * frame_height)
                            cv2.circle(display_frame, (px, py), 3, (0, 0, 255), -1)
                    
                    # Process right hand if detected
                    if right_hand is not None:
                        cv2.drawContours(display_frame, [right_hand], 0, (0, 255, 0), 2)
                        cv2.putText(display_frame, "Right (CV)", 
                                   (frame_width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Estimate keypoints
                        keypoints = estimate_hand_keypoints(right_hand, False, (frame_height, frame_width))
                        
                        # Store keypoints in data
                        for x, y, idx in keypoints:
                            frame_data[f"Right_Point{idx}_x"] = x
                            frame_data[f"Right_Point{idx}_y"] = y
                            
                            # Draw keypoints
                            px, py = int(x * frame_width), int(y * frame_height)
                            cv2.circle(display_frame, (px, py), 3, (0, 255, 0), -1)
                else:
                    print(f"🚫 No hands detected in Frame {frame_count}")
                
                # Show skin mask for debugging
                skin_mask_display = cv2.resize(skin_mask, (320, 240))
                skin_mask_color = cv2.cvtColor(skin_mask_display, cv2.COLOR_GRAY2BGR)
                display_frame[frame_height-250:frame_height-10, 10:330] = skin_mask_color
                
            except Exception as e:
                print(f"⚠️ OpenCV error: {str(e)}")
        
        # Add frame number and detection method to video
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        detection_method = "MediaPipe" if mediapipe_detected else "OpenCV" if opencv_detections > mediapipe_detections else "None"
        cv2.putText(display_frame, f"Method: {detection_method}", (10, frame_height - 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Append data
        data.append(frame_data)
        
        # Write to output video
        out.write(display_frame)
        
        # Display frame
        resized_frame = cv2.resize(display_frame, (min(frame_width, 1280), min(frame_height, 720)))
        cv2.imshow("Hand Tracking", resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    
    # Calculate statistics
    mediapipe_rate = mediapipe_detections / frame_count * 100 if frame_count > 0 else 0
    opencv_rate = opencv_detections / frame_count * 100 if frame_count > 0 else 0
    total_rate = (mediapipe_detections + opencv_detections) / frame_count * 100 if frame_count > 0 else 0
    
    print(f"\n📊 DETECTION STATISTICS:")
    print(f"  - Total frames processed: {frame_count}")
    print(f"  - MediaPipe detections: {mediapipe_detections} ({mediapipe_rate:.1f}%)")
    print(f"  - OpenCV detections: {opencv_detections} ({opencv_rate:.1f}%)")
    print(f"  - Total detection rate: {total_rate:.1f}%")
    print(f"\n✅ Annotated video saved at: {OUTPUT_VIDEO_PATH}")
    print(f"✅ Keypoints saved at: {CSV_OUTPUT_PATH}")

# Run the processing
if __name__ == "__main__":
    try:
        process_video()
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")