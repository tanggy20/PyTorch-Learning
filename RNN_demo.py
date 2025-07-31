import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

time_steps = np.linspace(0, 40, 1000)
data = np.sin(time_steps) + 0.3 * np.random.randn(1000)

seq_len = 20
X, y = [], [] 
for i in range(len(data) - seq_len):
    X.append(data[i:i + seq_len])
    y.append(data[i + seq_len])
X = np.array(X)
y = np.array(y)
X = torch.tensor(X,dtype=torch.float32).unsqueeze(2)  # Shape: (batch_size, seq_len, 1)
y = torch.tensor(y,dtype=torch.float32).unsqueeze(1)  # Shape: (batch_size, 1)
print("X shape:", X.shape, "y shape:", y.shape)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=1, 
            hidden_size=50, 
            num_layers=1, 
            batch_first=True
        )
        self.out = nn.Linear(50, 1)
    def forward(self, x ,hidden):
        r_out, hidden = self.rnn(x, hidden)
        out = self.out(r_out[:, -1, :])
        return out, hidden

model = RNNModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

EPOCHS = 200
BATCH_SIZE = 32

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_loss_list = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        hidden = None
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {train_loss / len(train_loader):.4f}')
    train_loss_list.append(train_loss / len(train_loader))

model.eval()
preds = []
with torch.no_grad():
    hidden = None
    pred, _ = model(X_test, hidden)
    preds = pred.numpy()

# 可视化：真实值 vs 预测值
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='training loss')
plt.subplot(1, 2, 2)
plt.plot(time_steps[train_size+seq_len:], y_test.numpy(), label='ground truth', alpha=0.7)
plt.plot(time_steps[train_size+seq_len:], preds, label='predicted value', linestyle='--')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.title('RNN time series prediction')
plt.show()