import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. 生成三分类数据集
n_data = torch.ones(100, 2)  # 每个类别100个样本，2个特征
# 三类数据分别围绕(2,2)、(-2,2)、(0,-2)分布
x0 = torch.normal(2 * n_data, 1)     # 类别0
y0 = torch.zeros(100)                # 标签0
x1 = torch.normal(torch.tensor([-2.0, 2.0]) * n_data, 1)  # 类别1
y1 = torch.ones(100)                 # 标签1
x2 = torch.normal(torch.tensor([0.0, -2.0]) * n_data, 1)  # 类别2
y2 = torch.full((100,), 2)           # 标签2
# 合并数据
x = torch.cat((x0, x1, x2), 0).type(torch.FloatTensor)  # 形状(300, 2)
y = torch.cat((y0, y1, y2), 0).type(torch.LongTensor)   # 形状(300,)，标签0/1/2

plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=y.numpy(), s=50, cmap='cool', alpha=0.7)
plt.title('Generated Data for Three Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
plt.pause(1.0)     

# 2. 定义三分类网络（输出层3个神经元）
class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)  # 输出层神经元数=类别数（3）
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 输出未归一化的分数（logits）
        return x

# 3. 初始化网络、损失函数和优化器
net = Net(2, 10, 3)  # 输入2特征，隐藏层10神经元，输出3类别
criterion = nn.CrossEntropyLoss()  # 交叉熵损失支持多分类
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# 4. 训练网络
plt.ion()  # 交互模式
for epoch in range(1, 101):
    # 前向传播
    outputs = net(x)  # 形状(300, 3)，每行对应3个类别的分数
    loss = criterion(outputs, y)  # 计算损失
    
    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每2轮可视化结果
    if epoch % 2 == 0:
        # 预测类别：取每行最大值的索引（0/1/2）
        predictions = torch.max(outputs, 1)[1]
        pred_y = predictions.numpy()
        target_y = y.numpy()
        
        # 计算准确率
        accuracy = float((pred_y == target_y).astype(int).sum()) / len(target_y)
        
        # 绘制散点图
        plt.cla()
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=50, cmap='cool', alpha=0.7)
        plt.text(-4, 4, f'Epoch: {epoch}, Loss: {loss.item():.4f}, Acc: {accuracy:.2f}', fontdict={'size': 12})
        plt.pause(0.1)

plt.ioff()
plt.show()
