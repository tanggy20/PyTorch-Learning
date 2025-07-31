import torch
import numpy as np
import torch.nn.functional as F 
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)
x_np = x.numpy()

y_relu = F.relu(x).numpy()
y_sigmoid = F.sigmoid(x).numpy()
y_tanh = F.tanh(x).numpy()
y_softplus = F.softplus(x).numpy()

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(x_np, y_relu, label='ReLU')
plt.title('ReLU Activation Function')
plt.subplot(2, 2, 2)
plt.plot(x_np, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.subplot(2, 2, 3)
plt.plot(x_np, y_tanh, label='Tanh')
plt.title('Tanh Activation Function')
plt.subplot(2, 2, 4)
plt.plot(x_np, y_softplus, label='Softplus')
plt.title('Softplus Activation Function')
plt.legend()
plt.tight_layout()
plt.show()



## Example of converting numpy array to torch tensor and back
# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()

# print(
#     '\n numpy:', np_data,
#     '\n torch:', torch_data,
#     '\n tensor2array:', tensor2array,
# )

# data = [[1,2], [3,4]]
# tensor = torch.FloatTensor(data)
# data = np.array(data)
# print('\n data:', data, '\n tensor:', tensor)
# print('\n numpy:', np.matmul(data, data),
#         '\n torch:', torch.mm(tensor, tensor))

# tensor = torch.tensor([[1,2], [3,4]],dtype = torch.float32,requires_grad=True)

# t_out = torch.mean(tensor * tensor)

# t_out.backward()
# print('t_out.grad:', tensor.grad)
# print(tensor)