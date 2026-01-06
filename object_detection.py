import cv2

# Load class labels MobileNet SSD was trained on
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor","book","phone","headphones"]

# Load pretrained model and config
# Ensure these filenames match the files in your directory
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Starting detection... Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Preprocess the frame: resize to 300x300 and create a blob
    # Parameters: image, scalefactor, size, mean subtraction
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                  0.007843, (300, 300), 127.5)

    # Set the input and run the forward pass
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence (probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring confidence is > 50%
        if confidence > 0.5:
            # Extract the index of the class label
            idx = int(detections[0, 0, i, 1])
            label = class_names[idx]

            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw the label and confidence percentage
            text = f"{label} ({int(confidence * 100)}%)"
            cv2.putText(frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Live Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()