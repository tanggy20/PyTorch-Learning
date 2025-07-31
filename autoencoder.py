import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch.utils.data as Data
from PIL import Image
import io

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5


train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
)

train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True,
)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 1, 28, 28)
        return decoded, encoded


autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
criterion = nn.MSELoss()

f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()


view_data = train_data.data[:N_TEST_IMG].view(-1, 28 * 28).float()/ 255.0
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

frames = []
for epoch in range(EPOCH):
    for step, (x, _) in enumerate(train_loader):
        x = x.float()
        optimizer.zero_grad()
        decoded, encoded = autoencoder(x)
        loss = criterion(decoded, x)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, EPOCH, step + 1, len(train_loader), loss.item()))
            
            decoded_img, _ = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_img.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            
            plt.draw()
            plt.pause(0.01)
            # 关键修改：用缓冲区获取图像数据，避免形状计算错误
            buf = io.BytesIO()  # 创建内存缓冲区
            f.savefig(buf, format='png', bbox_inches='tight')  # 保存当前Figure到缓冲区
            buf.seek(0)  # 移动到缓冲区开头
            img = Image.open(buf)  # 读取图像
            frames.append(img.copy())  # 复制图像到帧列表（避免缓冲区释放问题）
            buf.close()  # 关闭缓冲区


plt.ioff()
plt.show()

#保存动态图为GIF
if frames:
    frames[0].save(
        'autoencoder_training.gif',
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 每帧显示200毫秒（5帧/秒）
        loop=0  # 无限循环
    )
    print("动态图已保存为 'autoencoder_training.gif'")
else:
    print("未捕获到有效帧，无法保存动态图")


view_data = train_data.data[:500].float().view(-1, 28 * 28) / 255.0
_, encoded_data = autoencoder(view_data)
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.grid()
X, Y, Z = encoded_data.data.numpy()[:,0], encoded_data.data.numpy()[:,1], encoded_data.data.numpy()[:,2]
values = train_data.targets[:500].numpy()
# for x, y, z, val in zip(X, Y, Z, values):
#     value = cm.rainbow(int(255*val/9))
#     ax.scatter(x, y, z, c=value, marker='o', s=10)
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)) 
    ax.text(x, y, z, s, backgroundcolor=c)
# ax.scatter(X, Y, Z, c=values, marker='o', s=10, cmap='rainbow')
# unique_values = np.unique(values)
# for val in unique_values:
#     indices = np.where(values == val)
#     color = cm.rainbow(int(255 * val / 9))
#     ax.scatter(X[indices], Y[indices], Z[indices],c = [color], marker='o', s=10, label=str(val))
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()