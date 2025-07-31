import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Hyperparameters
BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
# print(PAINT_POINTS.shape)s
# show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()

def artist_works():
    a = np.random.uniform(1, 2, size = BATCH_SIZE)[:,np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return paintings


G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),
)

D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)

optimizer_G = optim.Adam(G.parameters(), lr=LR_G)
optimizer_D = optim.Adam(D.parameters(), lr=LR_D)

plt.ion()
frames = []
for step in range(10000):
    print('Step:', step)
    artist_paintings = artist_works()
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)
    G_paintings = G(G_ideas)
    prob_artist1 = D(G_paintings)
    G_loss = - torch.mean(torch.log(prob_artist1 + 1e-8))
    optimizer_G.zero_grad()
    G_loss.backward()
    optimizer_G.step()

    prob_artist0 = D(artist_paintings)
    prob_G = D(G_paintings.detach())
    D_loss = - torch.mean(torch.log(prob_artist0 + 1e-8) + torch.log(1 - prob_G + 1e-8))
    optimizer_D.zero_grad()
    D_loss.backward()
    optimizer_D.step()

   
    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings[0].detach().numpy(), c='#4AD631', lw=3, label='Generated painting')
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (1.38 for G to converge)' % D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=10)
        plt.draw()
        plt.pause(0.01)
        buf = io.BytesIO()  # 创建内存缓冲区
        plt.savefig(buf, format='png', bbox_inches='tight')  # 保存当前Figure到缓冲区
        buf.seek(0)  # 移动到缓冲区开头
        img = Image.open(buf)  # 读取图像
        frames.append(img.copy())  # 复制图像到帧列表（避免缓冲区释放问题）
        buf.close()  # 关闭缓冲区

plt.ioff()
plt.show()

if frames:
    frames[0].save(
        'GAN.gif',
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 每帧显示200毫秒（5帧/秒）
        loop=0  # 无限循环
    )
    print("动态图已保存为 'GAN.gif'")
else:
    print("未捕获到有效帧，无法保存动态图")