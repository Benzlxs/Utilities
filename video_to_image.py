import cv2
import os

# Path to the video file
video_path = '/home/li325/projects/Track-Anything/result/track/keyboardx225.mp4'

# Directory where images will be saved
output_dir = video_path.replace(".mp4", "_img")
os.makedirs(output_dir, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_num = 0
while True:
    # Read frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        break

    # Save frame as JPEG file
    frame_filename = os.path.join(output_dir, f"{frame_num:05d}.png")
    cv2.imwrite(frame_filename, frame)
    frame_num += 1

# When everything done, release the capture
cap.release()
print(f"Extracted {frame_num} frames to {output_dir}")

