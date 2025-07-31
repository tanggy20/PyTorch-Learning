import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision

# hyper parameters
EPOCH = 3
BATCH_SIZE = 50
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

train_loader = torch.utils.data.DataLoader(
    dataset = train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        r_out, _ = self.rnn(x, (h0, c0))
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

plt.ion()
train_loss_list = []
test_loss_list = []
accuracy_list = []

for epoch in range(EPOCH):
    rnn.train()
    train_loss = 0.0
    for step, (train_img, train_label) in enumerate(train_loader):
        train_img = train_img.view(-1, TIME_STEP, INPUT_SIZE)
        output = rnn(train_img)
        loss = criterion(output, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if step % 50 == 0:
            rnn.eval()
            correct = 0
            total = 0
            test_loss = 0.0
            with torch.no_grad():
                for test_img, test_label in test_loader:
                    test_img = test_img.view(-1, TIME_STEP, INPUT_SIZE)
                    test_output = rnn(test_img)
                    test_loss += criterion(test_output, test_label).item()
                    _, predicted = torch.max(test_output.data, 1)
                    total += test_label.size(0)
                    correct += (predicted == test_label).sum().item()

            accuracy = correct / total * 100
            train_loss_list.append(train_loss / (step + 1))
            test_loss_list.append(test_loss / len(test_loader))
            accuracy_list.append(accuracy)
            print(f'Epoch [{epoch+1}/{EPOCH}], Step [{step+1}/{len(train_loader)}], '
                  f'Train Loss: {train_loss / (step + 1):.4f}, '
                  f'Test Loss: {test_loss / len(test_loader):.4f}, '
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

