import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data

from PIL import Image
import io

#Hyperparameters
N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCHS = 12
LR = 0.03
N_HIDDEN = 8
ACTIVATION = F.tanh
B_INIT = -0.2

# training data
x = torch.linspace(-7, 10, N_SAMPLES).view(-1, 1)
noise = torch.randn(N_SAMPLES, 1) * 2.0
y = x.pow(2) + noise - 5

# test data
x_test = torch.linspace(-7, 10, 200).view(-1, 1)
noise_test = torch.randn(200, 1) * 2.0
y_test = x_test.pow(2) + noise_test - 5

# plt.scatter(x.numpy(), y.numpy(), s = 5, label='Training Data')
# plt.scatter(x_test.numpy(), y_test.numpy(), s = 5, label='Test Data')
# plt.legend()
# plt.show()

train_dataset = Data.TensorDataset(x, y)
train_loader = Data.DataLoader(dataset = train_dataset, batch_size=BATCH_SIZE, shuffle=True)



class Net(nn.Module):
    def __init__(self,batch_normalization):
        super(Net,self).__init__()
        self.do_bn = batch_normalization
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)

        for i in range(N_HIDDEN):
            input_size = 1 if i == 0 else 10
            fc = nn.Linear(input_size, 10)
            self._set_init(fc)
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)
                self.bns.append(bn)
        self.output = nn.Linear(10, 1)
        self._set_init(self.output) 

    def _set_init(self,layer):
        nn.init.normal_(layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(layer.bias, B_INIT)
    
    def forward(self, x):
        pre_activation = [x]
        if self.do_bn:
            x = self.bn_input(x)
        layer_input = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            if self.do_bn:
                x = self.bns[i](x)
            pre_activation.append(x)
            x = ACTIVATION(x)
            layer_input.append(x)
        x = self.output(x)
        return x, pre_activation, layer_input

nets = [Net(batch_normalization) for batch_normalization in [False, True]]
# print(*nets, sep='\n')
optimizers = [optim.Adam(net.parameters(), lr=LR) for net in nets]
criterion = nn.MSELoss()


def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        # print(pre_ac[8])
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0:
            p_range = (-7, 10)
            the_range = (-7, 10)
        else:
            p_range = (-10, 10)
            the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359',alpha=0.5,edgecolor='black',linewidth=1.2)
        ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5,edgecolor='black',linewidth=1.2)
        ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359',edgecolor='black',linewidth=1.2)
        ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF',edgecolor='black',linewidth=1.2)
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]: 
            a.set_yticks(())
            a.set_xticks(())
        ax_pa_bn.set_xticks(p_range)
        ax_bn.set_xticks(the_range)
        axs[0, 0].set_ylabel('PreAct')
        axs[1, 0].set_ylabel('BN PreAct')
        axs[2, 0].set_ylabel('Act')
        axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)
    buf = io.BytesIO()  # 创建内存缓冲区
    f.savefig(buf, format='png', bbox_inches='tight')  # 保存当前Figure到缓冲区
    buf.seek(0)  # 移动到缓冲区开头
    img = Image.open(buf)  # 读取图像
    frames.append(img.copy())  # 复制图像到帧列表（避免缓冲区释放问题）
    buf.close()  # 关闭缓冲区

if __name__ == "__main__":
    f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))
    plt.ion()
    plt.show()

    losses = [[],[]]
    frames = []
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        layer_inputs = []
        pre_activations = []
        for net in nets:
            net.eval()
            output, pre_activation, layer_input = net(x_test)
            pre_activations.append(pre_activation)
            layer_inputs.append(layer_input)
        plot_histogram(*layer_inputs, *pre_activations)
        for i, (b_x, b_y) in enumerate(train_loader):
            for net, optimizer in zip(nets, optimizers):
                net.train()
                optimizer.zero_grad()
                output, _, _ = net(b_x)
                loss = criterion(output, b_y)
                loss.backward()
                optimizer.step()
                losses[nets.index(net)].append(loss.item())
            
plt.ioff()

if frames:
    frames[0].save(
        'Batch_Normalization_tanh.gif',
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 每帧显示200毫秒（5帧/秒）
        loop=0  # 无限循环
    )
    print("动态图已保存为 'Batch_Normalization_tanh.gif'")
else:
    print("未捕获到有效帧，无法保存动态图")
# plot training loss
plt.figure(2)
plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
plt.plot(losses[1], c="#74FFB0", lw=3, label='Batch Normalization')
plt.xlabel('step');plt.ylabel('test loss');plt.ylim((0, 2000));plt.legend(loc='best')


 # evaluation
# set net to eval mode to freeze the parameters in batch normalization layers
[net.eval() for net in nets]    # set eval mode to fix moving_mean and moving_var
preds = [net(x_test)[0] for net in nets]
plt.figure(3)
plt.plot(x_test.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
plt.plot(x_test.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
plt.scatter(x_test.data.numpy(), y_test.data.numpy(), c='r', s=50, alpha=0.2, label='test data')
plt.legend(loc='best')
plt.show()