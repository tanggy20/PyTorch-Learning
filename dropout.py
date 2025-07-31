import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import io
from PIL import Image

# torch.manual_seed(1)  # For reproducibility

#Hyper Parameters
N_SAMPLES = 20
N_HIDDEN = 50

# train data
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), dim=1)
y = x + 0.3 * torch.randn(x.size())

# test data

x_test = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), dim=1)
y_test = x_test + 0.3 * torch.randn(x_test.size())

# plt.scatter(x.numpy(), y.numpy(),c='magenta', label='train data')
# plt.scatter(x_test.numpy(), y_test.numpy(),c='cyan', label='test data')
# plt.legend(loc='upper left')
# plt.show()

class DropNet(nn.Module):
    def __init__(self):
        super(DropNet, self).__init__()
        self.hidden = nn.Linear(1, N_HIDDEN)
        self.predict = nn.Linear(N_HIDDEN, N_HIDDEN)
        self.output = nn.Linear(N_HIDDEN, 1)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        x = torch.relu(self.predict(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

overfittingnet = nn.Sequential(
    nn.Linear(1, N_HIDDEN),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, N_HIDDEN),
    nn.ReLU(),
    nn.Linear(N_HIDDEN, 1)
)
dropnet = DropNet()

optimizer_overfit = torch.optim.Adam(overfittingnet.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(dropnet.parameters(), lr=0.01)
criterion = nn.MSELoss()

plt.ion()
frames = []
for epoch in range(1000):
    # Overfitting network
    optimizer_overfit.zero_grad()
    output_overfit = overfittingnet(x)
    loss_overfit = criterion(output_overfit, y)
    loss_overfit.backward()
    optimizer_overfit.step()

    # Dropout network
    optimizer_drop.zero_grad()
    output_drop = dropnet(x)
    loss_drop = criterion(output_drop, y)
    loss_drop.backward()
    optimizer_drop.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss Overfit: {loss_overfit.item():.4f}, Loss Dropout: {loss_drop.item():.4f}')
        overfittingnet.eval()
        dropnet.eval()

        plt.cla()
        test_pred_overfit = overfittingnet(x_test)
        test_pred_drop = dropnet(x_test)
        plt.scatter(x.numpy(), y.numpy(), c='magenta', label='Train Data')
        plt.scatter(x_test.numpy(), y_test.numpy(), c='cyan', label='Test Data')
        plt.plot(x_test.numpy(), test_pred_overfit.detach().numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(x_test.numpy(), test_pred_drop.detach().numpy(), 'g-', lw=3, label='Dropout(50%)')
        plt.text(0, -0.5, 'overfitting loss=%.4f' % criterion(test_pred_overfit, y_test).data.numpy(), fontdict={'size': 16, 'color':  'red'})
        plt.text(0, -1.0, 'dropout loss=%.4f' % criterion(test_pred_drop, y_test).data.numpy(), fontdict={'size': 16, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.pause(0.1)
        buf = io.BytesIO()  # 创建内存缓冲区
        plt.savefig(buf, format='png', bbox_inches='tight')  # 保存当前Figure到缓冲区
        buf.seek(0)  # 移动到缓冲区开头
        img = Image.open(buf)  # 读取图像
        frames.append(img.copy())  # 复制图像到帧列表（避免缓冲区释放问题）
        buf.close()  # 关闭缓冲区
        overfittingnet.train()
        dropnet.train()

plt.ioff()
plt.show()
if frames:
    frames[0].save(
        'dropout.gif',
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 每帧显示200毫秒（5帧/秒）
        loop=0  # 无限循环
    )
    print("动态图已保存为 'dropout.gif'")
else:
    print("未捕获到有效帧，无法保存动态图")
