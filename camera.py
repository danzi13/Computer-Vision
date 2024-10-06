import cv2
import os

def create_photos_folder():
    """Create a photos folder if it doesn't exist."""
    folder_name = "photos"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def capture_photo(card_name, count, folder_name, frame):
    """Capture a photo and save it with the appropriate name."""
    photo_name = f"{card_name}-{count:02d}.png"
    photo_path = os.path.join(folder_name, photo_name)
    cv2.imwrite(photo_path, frame)
    print(f"Photo saved as {photo_name}")

def main():
    # Create photos folder
    folder_name = create_photos_folder()

    # Open a connection to the webcam (use 0 for default camera, 1 for another, or IP stream)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Instructions:")
    print("s - Snapshot")
    print("esc - Quit program")

    while True:
        # Ask the user for the card name
        card_name = input("Enter the card name (e.g., KS for King of Spades): ")
        count = 1

        while count <= 5:  # Capture 5 photos for each card
            # Capture frame-by-frame from webcam
            retval, img = cap.read()
            if not retval:
                print("Failed to grab frame")
                break

            # Rescale the input image if desired
            res_scale = 0.5
            img_resized = cv2.resize(img, (0, 0), fx=res_scale, fy=res_scale)

            # Display the frame
            cv2.imshow("Live Camera", img_resized)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # Take a snapshot when 's' is pressed
                capture_photo(card_name, count, folder_name, img)
                count += 1  # Increment count for the next photo
            elif key == 27:  # Exit when 'esc' is pressed
                cap.release()
                cv2.destroyAllWindows()
                print("Program exited.")
                return

        # After 5 photos, ask for the next card
        print(f"Finished taking 5 photos for {card_name}. Ready for the next card!")

    # Release the camera and close windows (this will be called on 'esc')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import os

def create_photos_folder():
    """Create a photos folder if it doesn't exist."""
    folder_name = "photos"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def capture_photo(card_name, count, folder_name, frame):
    """Capture a photo and save it with the appropriate name."""
    photo_name = f"{card_name}-{count:02d}.png"
    photo_path = os.path.join(folder_name, photo_name)
    cv2.imwrite(photo_path, frame)
    print(f"Photo saved as {photo_name}")

def main():
    # Create photos folder
    folder_name = create_photos_folder()

    # Open a connection to the webcam (use 0 for default camera, 1 for another, or IP stream)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Instructions:")
    print("s - Snapshot")
    print("esc - Quit program")

    while True:
        # Ask the user for the card name
        card_name = input("Enter the card name (e.g., KS for King of Spades): ")
        count = 1

        while count <= 5:  # Capture 5 photos for each card
            # Capture frame-by-frame from webcam
            retval, img = cap.read()
            if not retval:
                print("Failed to grab frame")
                break

            # Rescale the input image if desired
            res_scale = 0.5
            img_resized = cv2.resize(img, (0, 0), fx=res_scale, fy=res_scale)

            # Display the frame
            cv2.imshow("Live Camera", img_resized)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # Take a snapshot when 's' is pressed
                capture_photo(card_name, count, folder_name, img)
                count += 1  # Increment count for the next photo
            elif key == 27:  # Exit when 'esc' is pressed
                cap.release()
                cv2.destroyAllWindows()
                print("Program exited.")
                return

        # After 5 photos, ask for the next card
        print(f"Finished taking 5 photos for {card_name}. Ready for the next card!")

    # Release the camera and close windows (this will be called on 'esc')
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
