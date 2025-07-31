import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

x = torch.linspace(-1, 1, 100)
x = x.unsqueeze(1)
y = x.pow(2) + 0.2 * torch.randn(x.size())
# plt.scatter(x.numpy(), y.numpy(), label='Data Points', alpha=0.5)
# plt.show()

class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net(1, 10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

plt.ion()
# plt.show()
# Training the model
for epoch in range(1,201):
    optimizer.zero_grad()
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        plt.cla()
        plt.scatter(x.numpy(), y.numpy(), label='Data Points', alpha=0.5)
        plt.plot(x.numpy(), outputs.detach().numpy(), 'r-', label='Model Prediction')
        plt.text(-0.5, 1.0, 'Epoch [{}/200], Loss: {:.4f}'.format(epoch, loss.item()), 
                 fontdict={'size': 10, 'color': 'red'})
        plt.legend()
        plt.pause(0.1)
plt.ioff()
plt.show()

