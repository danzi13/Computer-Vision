import torch
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import cv2
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained SSD model and weights
model = ssdlite320_mobilenet_v3_large(weights='SSDLite320_MobileNet_V3_Large_Weights.COCO_V1')
model.load_state_dict(torch.load('ssd_card_model.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Transformation for the model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 320))
])

# Label mappings
label_mapping_value = {
    0: "A", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8",
    8: "9", 9: "10", 10: "J", 11: "Q", 12: "K"
}

label_mapping_suit = {
    0: "C", 1: "D", 2: "H", 3: "S"
}

# Function to crop the largest detected card from the frame
def crop_card_from_frame(frame):
    """
    Crop the card directly from the current frame.
    """
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 2: Apply Sobel operator to find edges
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(sobel_combined)

    # Step 3: Threshold the Sobel result to create a binary image
    _, binary_image = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)

    # Step 4: Find contours based on the binary Sobel edge map
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Identify the largest contour, assuming it's the card
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Step 6: Calculate the bounding rectangle for cropping
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Step 7: Crop the card from the original image
        cropped_card = frame[y:y + h, x:x + w]
        return cropped_card
    else:
        return None

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use the default webcam
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 'Spacebar' to evaluate the current frame.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the live feed
    cv2.imshow("Live Card Detection", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):  # Spacebar pressed
        print("Processing snapshot...")
        cropped_card_image = crop_card_from_frame(frame)

        if cropped_card_image is not None:
            # Pass the cropped card into the model
            model_input = transform(cropped_card_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(model_input)

            # Process predictions with confidence > 0.1
            if len(outputs[0]['scores']) > 0:
                max_idx = torch.argmax(outputs[0]['scores'])
                score = outputs[0]['scores'][max_idx].cpu().item()
                label_idx = int(outputs[0]['labels'][max_idx].cpu().item())

                if score > 0.1:
                    # Map label index to value and suit
                    label_value = label_mapping_value.get(label_idx % 13, "Unknown")
                    label_suit = label_mapping_suit.get(label_idx // 13, "Unknown")
                    label = f"{label_value} of {label_suit} ({score:.2f})"

                    # Add prediction label to the image
                    cv2.putText(cropped_card_image, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    print(f"Detected: {label}")
                else:
                    print("No confident prediction.")
            else:
                print("No predictions made by the model.")

            # Display the processed cropped card with prediction
            cv2.imshow("Detected Card", cropped_card_image)

        else:
            print("No card detected in the frame.")

# Release resources
cap.release()
cv2.destroyAllWindows()
