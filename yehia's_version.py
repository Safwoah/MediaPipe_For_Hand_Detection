import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe Hands with more reliable settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # Process each frame independently
    max_num_hands=2,
    model_complexity=1,       # Higher complexity for better accuracy
    min_detection_confidence=0.3,  # Low threshold to catch difficult hands
    min_tracking_confidence=0.3
)
mp_draw = mp.solutions.drawing_utils

# Input & Output Paths
video_path = r"D:\Subject #1\Day 1\Before\IMG_2982.MOV"  # Change to your video file
output_video_path = "Day15_after_annotated.mp4"
csv_output_path = "Day15_after.csv"

# Open video
cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print(f"❌ ERROR: Could not open video {video_path}")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Check FPS value
if fps == 0:
    print("⚠️ WARNING: FPS is 0, setting default 30 FPS")
    fps = 30

# Video Writer (to save annotated output)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Data storage
data = []
frame_count = 0

print("✅ Processing video...")

# Define a simple function to enhance image for better hand detection
def enhance_image(image):
    # Apply contrast enhancement
    enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=20)
    return enhanced

# Optional: Define a function to split image for left/right processing
def process_split_image(rgb_frame):
    # Split the image into left and right halves
    height, width = rgb_frame.shape[:2]
    left_half = rgb_frame[:, 0:width//2]
    right_half = rgb_frame[:, width//2:]
    
    # Process each half separately
    left_result = hands.process(enhance_image(left_half))
    right_result = hands.process(enhance_image(right_half))
    
    # Create combined results (placeholder - actual implementation would be more complex)
    combined_landmarks = []
    combined_handedness = []
    
    # Process left half results
    if left_result.multi_hand_landmarks:
        # Need to adjust x coordinates to be relative to original image
        for landmarks in left_result.multi_hand_landmarks:
            # Scale x coordinates back to full image
            for landmark in landmarks.landmark:
                landmark.x = landmark.x / 2  # Scale to half the width
            combined_landmarks.append(landmarks)
        
        # Add handedness info and force "Left" label
        for handedness in left_result.multi_handedness:
            handedness.classification[0].label = "Left"
            combined_handedness.append(handedness)
    
    # Process right half results
    if right_result.multi_hand_landmarks:
        # Need to adjust x coordinates to be relative to original image
        for landmarks in right_result.multi_hand_landmarks:
            # Scale and shift x coordinates to map to right half of image
            for landmark in landmarks.landmark:
                landmark.x = 0.5 + (landmark.x / 2)  # Shift by 0.5 and scale
            combined_landmarks.append(landmarks)
        
        # Add handedness info and force "Right" label
        for handedness in right_result.multi_handedness:
            handedness.classification[0].label = "Right"
            combined_handedness.append(handedness)
    
    # Create a custom result object to return
    class Result:
        pass
    
    result = Result()
    result.multi_hand_landmarks = combined_landmarks if combined_landmarks else None
    result.multi_handedness = combined_handedness if combined_handedness else None
    
    return result

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete.")
        break  # Stop when video ends

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply image enhancement
    enhanced_frame = enhance_image(rgb_frame)
    
    # First approach: Try processing the whole frame
    result = hands.process(enhanced_frame)
    
    # If we detect fewer than 2 hands, try split-image approach
    if result.multi_hand_landmarks is None or len(result.multi_hand_landmarks) < 2:
        print(f"💡 Trying split-image approach for Frame {frame_count}")
        result = process_split_image(rgb_frame)

    # Create a copy of the frame for visualization
    display_frame = frame.copy()
    
    # Initialize frame data
    frame_data = {"Frame": frame_count}
    
    # Initialize columns for both hands even if not detected
    for hand_label in ["Left", "Right"]:
        for i in range(21):  # MediaPipe tracks 21 landmarks per hand
            frame_data[f"{hand_label}_Point{i}_x"] = None
            frame_data[f"{hand_label}_Point{i}_y"] = None
    
    # Track which hands were detected
    left_hand_processed = False
    right_hand_processed = False

    # Process detected hands
    if result.multi_hand_landmarks and result.multi_handedness:
        num_hands = len(result.multi_hand_landmarks)
        print(f"🖐️ {num_hands} hand(s) detected in Frame {frame_count}")

        for idx, (hand_landmarks, hand_info) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
            # Calculate average position to determine left/right in image
            avg_x = sum(landmark.x for landmark in hand_landmarks.landmark) / len(hand_landmarks.landmark)
            is_left_side = avg_x < 0.5
            
            # Get detected label and use position to override if necessary
            detected_label = hand_info.classification[0].label
            confidence = hand_info.classification[0].score
            
            # Make hand label consistent with what we see in the image
            # For this specific video, we know left side is Left hand 
            # and right side is Right hand based on your screenshot
            hand_label = "Left" if is_left_side else "Right"
            
            # Update tracking
            if hand_label == "Left":
                left_hand_processed = True
            else:
                right_hand_processed = True
            
            print(f"  - Hand {idx}: {hand_label} (confidence: {confidence:.2f})")
            
            # Add visual label on frame
            wrist_x = int(hand_landmarks.landmark[0].x * frame_width)
            wrist_y = int(hand_landmarks.landmark[0].y * frame_height)
            cv2.putText(display_frame, f"{hand_label}", (wrist_x, wrist_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Store all landmarks for this hand
            for i, landmark in enumerate(hand_landmarks.landmark):
                # Convert normalized coordinates to pixel values
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                # Store in frame data
                frame_data[f"{hand_label}_Point{i}_x"] = x
                frame_data[f"{hand_label}_Point{i}_y"] = y 
            
            # Draw hand landmarks with different colors based on hand
            color = (0, 255, 0) if hand_label == "Right" else (255, 0, 0)
            mp_draw.draw_landmarks(
                display_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
    
    # Add missing hand indicators
    if not left_hand_processed:
        cv2.putText(display_frame, "Left hand not detected", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if not right_hand_processed:
        cv2.putText(display_frame, "Right hand not detected", (frame_width - 300, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add frame number to video
    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Count hands detected
    hands_detected = int(left_hand_processed) + int(right_hand_processed)
    cv2.putText(display_frame, f"Hands detected: {hands_detected}/2", (10, frame_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Append frame data
    data.append(frame_data)

    # Write frame to output video
    out.write(display_frame)

    # Show real-time processing (optional)
    display_frame_resized = cv2.resize(display_frame, (min(frame_width, 1280), min(frame_height, 720)))
    cv2.imshow("Annotated Video", display_frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save keypoints data to CSV
df = pd.DataFrame(data)
df.to_csv(csv_output_path, index=False)

print(f"✅ Annotated video saved at: {output_video_path}")
print(f"✅ Keypoints saved at: {csv_output_path}")
print(f"✅ Processed {frame_count} frames")





"""
What i had to change in the code:

1- Static Image Mode:

Original: static_image_mode=False (using tracking between frames)
New version: static_image_mode=True (processing each frame independently)


2- Detection Confidence Thresholds:

Original: min_detection_confidence=0.5, min_tracking_confidence=0.3
New version: min_detection_confidence=0.3, min_tracking_confidence=0.3


3- Image Enhancement:

Original: No image enhancement
New version: Added enhance_image() function to improve contrast and brightness


4- Split Image Processing:

Original: No split image processing
New version: Added process_split_image() function to process left and right sides separately when regular detection fails


5- Hand Labeling Logic:

Original: Swapped labels with hand_label = "Left" if raw_label == "Right" else "Right"
New version: Uses position-based labeling with hand_label = "Left" if is_left_side else "Right"


6- Multiple Detection Methods:

Original: Just one detection attempt
New version: Two-stage process with fallback to split-image approach


7- Display Frame Handling:

Original: Modifies the original frame directly
New version: Creates a copy with display_frame = frame.copy() to preserve the original


8- Visual Indicators:

Original: Basic indicators for detection
New version: Added hand detection counter and more detailed feedback


9- Clearer Logic Organization:

New version has a more structured approach to processing detected hands and tracking which ones were detected
"""