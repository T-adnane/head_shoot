import cv2
import mediapipe as mp

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to draw a vertical and horizontal line through the midpoint between left and right eye keypoints
def draw_lines(image, landmarks):
    if landmarks.landmark:
        # Get left and right eye keypoints
        left_eye = (int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].x * image.shape[1]),
                    int(landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].y * image.shape[0]))
        right_eye = (int(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].x * image.shape[1]),
                     int(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].y * image.shape[0]))
        # Calculate the midpoint between left and right eye keypoints
        midpoint = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        # Draw vertical line at the midpoint
        cv2.line(image, (midpoint[0], 0), (midpoint[0], image.shape[0]), (255, 255, 255), 1)
        # Draw horizontal line through the midpoint with the same length as the width of the frame
        cv2.line(image, (0, midpoint[1]), (image.shape[1], midpoint[1]), (255, 255, 255), 1)

# Function to process frame and draw lines
def process_frame(image, pose):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image with MediaPipe Pose model
    results = pose.process(image_rgb)
    # Draw lines on the image
    if results.pose_landmarks:
        draw_lines(image, results.pose_landmarks)
    return image

# Main function
def main():
    # Initialize MediaPipe Pose model
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Open video capture
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            # Read a frame from the video capture
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame = process_frame(frame, pose)
            processed_frame = cv2.flip(processed_frame, 1)


            # Display the processed frame
            cv2.imshow('Pose Estimation', processed_frame)

            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture
        cap.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
