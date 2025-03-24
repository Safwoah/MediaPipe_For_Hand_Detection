import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe Hands with improved settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# Input & Output Paths
video_path = r"D:\Tremor Videos Subject #1\Day15-After_rotated.mp4" # Change to your video file
output_video_path = "Day15-After_annotated.mp4"
csv_output_path = "Day15-After.csv"

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete.")
        break  # Stop when video ends

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb_frame)

    frame_data = {"Frame": frame_count}  # Store keypoints per frame
    
    # Initialize columns for both hands even if not detected
    # This ensures consistent CSV columns
    for hand_label in ["Left", "Right"]:
        for i in range(21):  # MediaPipe tracks 21 landmarks per hand
            frame_data[f"{hand_label}_Point{i}_x"] = None
            frame_data[f"{hand_label}_Point{i}_y"] = None
            

    if result.multi_hand_landmarks and result.multi_handedness:
        num_hands = len(result.multi_hand_landmarks)
        print(f"🖐️ {num_hands} hand(s) detected in Frame {frame_count}")

        for idx, (hand_landmarks, hand_info) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
            # Get hand label (Left or Right)
            raw_label = hand_info.classification[0].label
            hand_label = "Left" if raw_label == "Right" else "Right"

            # Display confidence for debugging
            confidence = hand_info.classification[0].score
            print(f"  - Hand {idx}: {hand_label} (confidence: {confidence:.2f})")
            
            # Add visual label on frame
            wrist_x = int(hand_landmarks.landmark[0].x * frame_width)
            wrist_y = int(hand_landmarks.landmark[0].y * frame_height)
            cv2.putText(frame, f"{hand_label}", (wrist_x, wrist_y - 10), 
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
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
    else:
        print(f"🚫 No hands detected in Frame {frame_count}")

    # Add frame number to video
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Append frame data
    data.append(frame_data)

    # Write frame to output video
    out.write(frame)

    # Show real-time processing (optional)
    # Resize for display if video is large
    display_frame = cv2.resize(frame, (min(frame_width, 1280), min(frame_height, 720)))
    cv2.imshow("Annotated Video", display_frame)
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
