import cv2
import numpy as np

# Load the Haar cascade for glare detection (modify path as necessary)
glare_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture (0 for default webcam, or use a video file path)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame.")
        break
    
    # Convert the frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to identify bright spots (glare)
    _, bright_spots = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    # Find contours of bright spots
    contours, _ = cv2.findContours(bright_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Fit a circle to the bright spot
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Reduce glare intensity inside the circle
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        dimmed_frame = cv2.addWeighted(frame, 1.0, mask, -0.5, 0)

        # Overlay the dimmed glare onto the original frame
        frame = np.where(mask > 0, dimmed_frame, frame)

    # Display the processed frame
    cv2.imshow("Glare Reduction", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
