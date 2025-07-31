import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#Hyper Parameters
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.01

steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)

# plt.plot(steps, x_np, 'r-', label='sin')
# plt.plot(steps, y_np,'g-', label='cos')
# plt.legend(loc = 'best')
# plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32,1)
    
    def forward(self,x,h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = self.out(r_out)
        return outs, h_state

rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
criterion = nn.MSELoss()


h_state = None  # Initialize hidden state

plt.figure(figsize=(10, 5))
plt.ion()
for step in range(60):
    start, end = step * np.pi, (step + 1) *np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)

    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    x, y = x.float(), y.float()

    optimizer.zero_grad()
    if h_state is not None:
        h_state = h_state.detach()

    outputs, h_state = rnn(x, h_state)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if step == 0:
        # 第一次绘制时添加label，用于生成图例
        plt.plot(steps, x_np, 'r-', label='input sin')
        plt.plot(steps, y_np, 'g-', label='target cos')
        plt.plot(steps, outputs.detach().numpy()[0, :, 0], 'b-', label='predicted cos')
    else:
        # 后续迭代不添加label，避免图例重复
        plt.plot(steps, x_np, 'r-')
        plt.plot(steps, y_np, 'g-')
        plt.plot(steps, outputs.detach().numpy()[0, :, 0], 'b-')
    print('step: {}, loss: {:.4f}'.format(step, loss.item()))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('RNN Regression')

    plt.pause(0.1)

plt.ioff()
plt.show()
