import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
import multiprocessing



if __name__ == '__main__':
    multiprocessing.freeze_support()  # 解决Windows多进程问题

    BATCH_SIZE = 5

    x = torch.linspace(1, 10, 10)
    y = torch.linspace(10, 1, 10)

    torch_dataset = Data.TensorDataset(x, y)
    Loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = BATCH_SIZE,
        shuffle= True,
        num_workers=2  # 多线程加载数据
    )

    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(Loader):
            print('Epoch:', epoch, '| Step:', step, '| batch_x:', batch_x.numpy(), '| batch_y:', batch_y.numpy())
            # 模拟训练过程
            # 这里可以添加模型的前向传播、损失计算和反向传播等步骤
            # 例如：outputs = model(batch_x)
            # loss = criterion(outputs, batch_y)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
