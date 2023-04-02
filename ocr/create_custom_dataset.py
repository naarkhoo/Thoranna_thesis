import os
from torchvision.transforms import Resize
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image


class CustomMNISTDataset(Dataset):
    def __init__(self, image_paths, labels, bounding_boxes, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.transform = transform

    def __getitem__(self, index):
        # Load the image and its corresponding label
        image = Image.open(self.image_paths[index]).convert('L')  # Convert to grayscale
        image_labels = self.labels[index]
        image_bboxes = self.bounding_boxes[index]

        digits = []
        digit_labels = []

        # Extract individual digits using the bounding boxes
        for i, bbox in enumerate(image_bboxes):
            x, y, w, h = bbox
            digit = image.crop((x, y, x + w, y + h))
            digit_label = image_labels[i]

            if self.transform:
                digit = self.transform(digit)

            digits.append(digit)
            digit_labels.append(digit_label)

        return digits, digit_labels

    def __len__(self):
        return len(self.image_paths)

def load_dataset():
    # Assuming you have a list of image file paths, corresponding labels (sequences), and bounding boxes for each image
    image_paths = ['IMG_2941.JPEG', 'IMG_2942.JPEG', 'IMG_2943.JPEG']
    sequence_labels = [[0, 2, 4, 5, 3], [0, 2, 4, 3, 3], [0, 2, 4, 2, 3]]  # Each element in this list should be a list of integers (the sequence of digits)
    bounding_boxes = [[[44, 48, 32, 37], [90, 40, 29, 47], [111, 42, 30, 45], [146, 40, 30, 50], [194, 39, 25, 61]],
                      [[65, 46, 34, 46], [95, 44, 30, 44], [124, 40, 22, 47], [153, 37, 24, 58], [20, 42, 37, 52]],
                      [[41, 53, 35, 46], [79, 53, 24, 41], [108, 49, 26, 46], [143, 44, 33, 57], [200, 42, 28, 65]]] # Each element in this list should be a list of bounding boxes (x, y, w, h) for each digit in the sequence

    # Define the transformations
    transform = Compose([
        Resize((28, 28)),          # Resize the image to 28x28 pixels
        ToTensor(),                # Convert the image to a PyTorch tensor
        Normalize((0.1307,), (0.3081,)),  # Normalize the image with the same values as the MNIST dataset
    ])

    # Create the custom dataset and DataLoader
    custom_train_data = CustomMNISTDataset(image_paths, sequence_labels, bounding_boxes, transform=transform)
    custom_train_loader = DataLoader(custom_train_data, batch_size=1, shuffle=True)  # Set the batch size to 1 to handle variable-length sequences
    return custom_train_loader


if __name__ == "__main__":
    load_dataset()
