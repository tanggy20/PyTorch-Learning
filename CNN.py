import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torchvision

# hyper parameters
EPOCH = 3
BATCH_SIZE = 50
LR = 0.01
DOWNLOAD_MNIST = False
DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
)

# print(train_data.data.size())
# print(train_data.targets.size())
# plt.imshow(train_data.data[0], cmap='gray')
# print(train_data.data[0].size())
# plt.title('%i' % train_data.targets[0])
# plt.show()

test_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train = False,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
)

train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size= BATCH_SIZE,
    shuffle = True,
)

test_loader = Data.DataLoader(
    dataset = test_data,
    batch_size= BATCH_SIZE,
    shuffle = False,
)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 7 * 7 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn = CNN().to(device=DEVICE)
# print(cnn)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

plt.ion()  # interactive mode on
train_loss_list = []
test_loss_list = []
accuracy_list = []

for epoch in range(EPOCH):
    cnn.train()
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()  # clear gradients for this training step
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


        if i % 50 == 0:
            cnn.eval()
            correct = 0
            total = 0
            test_loss = 0.0
            with torch.no_grad():
                for test_images, test_labels in test_loader:
                    test_images, test_labels = test_images.to(DEVICE), test_labels.to(DEVICE) 
                    test_outputs = cnn(test_images)
                    batch_loss = criterion(test_outputs, test_labels)
                    test_loss += batch_loss.item()

                    _, predicted = torch.max(test_outputs,1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            
            accuracy = correct / total * 100
            avg_train_loss = train_loss / (i+1)
            avg_test_loss = test_loss / len(test_loader)

            train_loss_list.append(avg_train_loss)
            test_loss_list.append(avg_test_loss)
            accuracy_list.append(accuracy)
            print(f'Epoch [{epoch+1}/{EPOCH}], Step [{i+1}/{len(train_loader)}], '
                  f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, '
                  f'Accuracy: {accuracy:.2f}%')
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.plot(train_loss_list, label='Train Loss')
            plt.plot(test_loss_list, label='Test Loss')
            plt.legend()
            plt.title('Loss Curve')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(accuracy_list, label='Accuracy')
            plt.legend()
            plt.ylabel('Accuracy')
            plt.xlabel('Steps')
            plt.title('Accuracy Curve')
            plt.pause(0.1)
            plt.tight_layout()

plt.ioff()  # interactive mode off
plt.show()  # show the plot at the end