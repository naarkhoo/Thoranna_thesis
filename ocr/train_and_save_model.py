import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from create_custom_dataset import load_dataset

"""
Model Architecture
--------------------------------------------------------------------------------

1. Convolutional Layer 1 (conv1): This layer takes a single 
channel input (grayscale image) and applies 32 filters (convolutional kernels)
of size 3x3 with a stride of 1 and padding of 1.
It generates 32 feature maps that capture different patterns in the input image.
Convolutional layers learn local features in the image.

2. ReLU Activation 1 (relu1): A Rectified Linear Unit (ReLU) activation
function is applied element-wise to the output of the first convolutional layer.
ReLU introduces non-linearity, which helps the network learn complex patterns.

3. Max Pooling Layer 1 (pool1): This layer reduces the spatial dimensions
of the input by taking the maximum value within non-overlapping 2x2 windows
and using a stride of 2. Pooling layers help to reduce computation, control overfitting,
and make the model more robust to small spatial variations in the input.

4. Convolutional Layer 2 (conv2): This layer takes the 32 feature
maps from the previous layer and applies 64 filters of size 3x3 with
a stride of 1 and padding of 1. It generates 64 feature maps that
capture more complex patterns in the input.

5. ReLU Activation 2 (relu2): A ReLU activation function is applied
to the output of the second convolutional layer.

6. Max Pooling Layer 2 (pool2): This layer reduces the spatial dimensions further, 
using the same max pooling operation as before in the first pooling layer.

7. Fully Connected Layer 1 (fc1): After the second max pooling layer, the output
is reshaped (flattened) into a 1D tensor and passed through a fully connected layer with 512 neurons.
Fully connected layers are used to learn global patterns from the features learned by the convolutional
and pooling layers.

8. ReLU Activation 3 (relu3): A ReLU activation function is applied
to the output of the first fully connected layer.

9. Fully Connected Layer 2 (fc2): The final fully connected layer has 10 neurons,
one for each of the possible digit classes (0-9). The output of this layer is used
to predict the class probabilities.

"""

# Define the CNN architecture
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=True)

    # Create the model, loss function and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}')
    
    custom_train_loader = load_dataset()
    # Fine-tune the model on the custom dataset
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_digits = 0

        for digits, digit_labels in custom_train_loader:
            for digit, label in zip(digits, digit_labels):
                digit, label = digit.to(device), torch.tensor([label], dtype=torch.long, device=device)

                optimizer.zero_grad()
                outputs = model(digit)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                total_digits += 1

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/total_digits:.4f}')

    # Test the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on the test set: {100 * correct / total:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), 'mnist_model.pth')