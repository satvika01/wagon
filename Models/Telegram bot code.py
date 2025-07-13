!pip install opencv-python
!pip install torch
!pip install telepot
!pip install ultralytics


import cv2
import telepot
from telepot.loop import MessageLoop
from ultralytics import YOLO

# Initialize the Telegram bot
token = 'your token'
bot = telepot.Bot(token)
chat_id = yourid

# Load the custom YOLOv8 model
model = YOLO("path to .pt file")

# Initialize the video capture
cap = cv2.VideoCapture(1)  # Default camera

# Initialize class colors
class_colors = {
    "closed_door": (0, 255, 0),  # Green
    "opened_door": (0, 0, 255),  # Red
    "wagon_number": (255, 0, 0),  # Blue
    "train": (0, 255, 255)  # Yellow
}

# Function to handle incoming Telegram messages
def handle_message(msg):
    global stop_detection
    command = msg['text'].lower()

    if command == "/stop":
        stop_detection = True
        bot.sendMessage(chat_id, "🛑 Detection stopped.")
        cap.release()
        cv2.destroyAllWindows()

# Start the bot message loop
MessageLoop(bot, handle_message).run_as_thread()
print("🚀 Bot is running. Send '/stop' to stop detection.")

# Start the detection loop
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read frame.")
            break

        frame = cv2.resize(frame, (1280, 720))
        results = model(frame)

        # Get detected objects
        detections = results[0].boxes.data  # Correct way to access bounding boxes

        # Filter detections for 'opened_door'
        opendoor_detections = [d for d in detections if int(d[5]) == 1]

        # Only send photo for 'opened_door' detections
        if opendoor_detections:
            print("🔹 Opened door detected! Sending photo...")

            # Draw bounding boxes for opened doors
            for det in opendoor_detections:
                x1, y1, x2, y2, conf, cls = det
                class_label = model.names[int(cls)]
                color = class_colors.get(class_label, (255, 255, 255))  # Default to white

                # Draw bounding box for opened door
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f'{class_label} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Save and send the frame only if an opened door is detected
            cv2.imwrite('opened_door_detected_frame.jpg', frame)
            bot.sendPhoto(chat_id, open('opened_door_detected_frame.jpg', 'rb'))

        # Draw bounding boxes for all detected objects
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            class_label = model.names[int(cls)]
            color = class_colors.get(class_label, (255, 255, 255))  # Default to white

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{class_label} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the frame
        cv2.imshow('Live Camera Feed', frame)

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🔴 Stopping detection...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Detection stopped successfully.")

else:
    print("❌ Error: Unable to access the camera.")
