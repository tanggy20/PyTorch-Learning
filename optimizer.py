import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

#hyper parameters

LR = 0.01
EPOCH = 12
BATCH_SIZE = 32


x = torch.unsqueeze(torch.linspace(-1,1, 1000), dim=1)
y = x.pow(2) + 0.2 * torch.randn(x.size())
# y = x.pow(2) + 0.2 * torch.normal(torch.zeros(*x.size()))
# print(y)

# plt.scatter(x.numpy(), y.numpy(), s=1)
# plt.show()

torch_dataset = torch.utils.data.TensorDataset(x, y)
Loader = torch.utils.data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":

    net_SGD = Net(1, 20, 1)
    net_Momentum = Net(1, 20, 1)
    net_RMSprop = Net(1, 20, 1)
    net_Adam = Net(1, 20, 1)
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    optimizer_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    optimizer_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    optimizer_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    optimizer_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99)) 
    optimizers = [optimizer_SGD, optimizer_Momentum, optimizer_RMSprop, optimizer_Adam]

    criterion = nn.MSELoss()
    loss_list = [[], [], [], []]

    for epoch in range(EPOCH):
        for batch, (batch_x, batch_y) in enumerate(Loader):
            for i, net in enumerate(nets):
                outputs = net(batch_x)
                loss = criterion(outputs, batch_y)
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
                loss_list[i].append(loss.item())
    
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    plt.figure(figsize=(10, 6))
    for i, loss in enumerate(loss_list):
        plt.plot(loss, label=labels[i])
    plt.title('Loss Curves for Different Optimizers')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
