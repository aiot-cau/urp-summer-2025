## Training a Classifier with PyTorch

The main steps are:

1.  **Load and Normalize Data**: Prepare the CIFAR-10 dataset for training and testing.
2.  **Define the Neural Network**: Build a Convolutional Neural Network (CNN).
3.  **Define Loss Function and Optimizer**: Set up the tools for training the network.
4.  **Train the Network**: Feed the training data to the model.
5.  **Test the Network**: Evaluate the model's performance on the test data.

-----

### 1. Load and Normalize CIFAR-10

First, we import the necessary libraries and prepare our data. `torchvision` makes it easy to load and apply transformations to datasets like CIFAR-10.

  - **Transformation**: The images are converted from PILImage format (with a pixel value range of [0, 1]) to Tensors and then normalized to a range of [-1, 1].
  - **Dataloaders**: The `DataLoader` manages batching, shuffling, and loading the data efficiently.

<!-- end list -->

```python
import torch
import torchvision
import torchvision.transforms as transforms

# Define transformations
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# Load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Define class labels
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

To verify the data, we can display a few random training images.

```python
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images and print labels
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
```

-----

### 2\. Define a Convolutional Neural Network (CNN)

Next, we define the architecture of our CNN. This network is designed for 3-channel (color) images and consists of:

  - Two convolutional layers (`conv1`, `conv2`) with ReLU activation and max-pooling.
  - Three fully-connected layers (`fc1`, `fc2`, `fc3`) that map the features to the final 10 classes.

<!-- end list -->

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

-----

### 3\. Define a Loss Function and Optimizer

For training, we need a loss function to measure the error and an optimizer to update the network's weights.

  - **Loss Function**: `CrossEntropyLoss` is suitable for multi-class classification.
  - **Optimizer**: `Stochastic Gradient Descent (SGD)` with momentum is used to adjust the weights.

<!-- end list -->

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

-----

### 4\. Train the Network

The training process involves iterating over the dataset multiple times (epochs). In each iteration, we:

1.  Get a batch of inputs and labels.
2.  Zero the gradients.
3.  Perform a forward pass to get predictions.
4.  Calculate the loss.
5.  Perform a backward pass to compute gradients.
6.  Update the weights using the optimizer.

<!-- end list -->

```python
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # calculate gradient
        optimizer.step() # update parameter(weight)

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

-----

### 5\. Test the Network on Test Data

After training, we evaluate the model's performance on the unseen test data.

First, we load the saved model and visualize a few test images with their ground-truth labels.

```python
# Load the model
net = Net()
net.load_state_dict(torch.load(PATH))

# Display a batch of test images
dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# Get model predictions
outputs = net(images)
_, predicted = torch.max(outputs, 1)    # 최댓값의 인덱스 계산
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
```

Next, we calculate the overall accuracy across the entire test set.

```python
correct = 0
total = 0
with torch.no_grad(): # Gradients are not needed for evaluation
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
```

Finally, we can analyze the accuracy for each individual class to see which ones the model learned best.

```python
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
```

