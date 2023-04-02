# ... (Keep the imports and class definition from the previous script)
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from train_and_save_model import MNISTClassifier
import subprocess

def convert_heic_to_jpg(heic_path, jpg_path):
    subprocess.run(['sips', '-s', 'format', 'jpeg', heic_path, '--out', jpg_path])

def main():
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTClassifier().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()

    # Load the image, convert to grayscale, and then threshold
    # convert_heic_to_jpg('IMG_2943.HEIC', 'IMG_2943.JPEG')
    image = cv2.imread('IMG_2945.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_digits = []

    for contour in contours:
        # Get the bounding box of each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding box on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the digit and resize to 28x28
        digit = gray[y:y+h, x:x+w]
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

        # Invert the digit and normalize the intensity values
        digit = 255 - digit
        digit = (digit / 255.0 - 0.1307) / 0.3081
        digit = digit.reshape(1, 1, 28, 28)

        # Convert to a PyTorch tensor and perform the prediction
        digit_tensor = torch.FloatTensor(digit).to(device)
        output = model(digit_tensor)
        _, prediction = torch.max(output.data, 1)

        # Add the detected digit and its coordinates to the list
        detected_digits.append((prediction.item(), x, y, x+w, y+h))

        # Draw the bounding box and prediction on the original image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, str(prediction.item()), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save the image with bounding boxes
    cv2.imwrite('image_with_bounding_boxes.jpg', image)

    # Print the detected digits and their bounding box coordinates
    print("Detected Digits:")
    for digit, x1, y1, x2, y2 in detected_digits:
        print(f"Digit: {digit}, Bounding Box: ({x1}, {y1}, {x2}, {y2})")

if __name__ == '__main__':
    main()