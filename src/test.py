import torch
import torch.cuda
import torch.nn as nn
import numpy as np
import importlib.util as importlib_util

# 是否开启dx_torch替代cuda
if importlib_util.find_spec(torch_directml):
    import torch_directml
    device = torch_directml.device()
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义一个线性回归模型类，它是继承自nn.Module的
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        # 调用父类的构造函数初始化
        super(LinearRegressionModel, self).__init__()
        # 创建一个线性层，它是一个简单的线性函数y=wx+b
        self.linear = nn.Linear(input_dim, output_dim)

    # 定义前向传播过程，输入x通过线性层得到输出
    def forward(self, x):
        out = self.linear(x)
        return out

def run():
    x_values = [i for i in range(11)]
    # 将列表转换为NumPy数组，并设置数据类型为float32
    x_train = np.array(x_values, dtype=np.float32)
    # 重塑数组，使其成为一列（特征），即(-1, 1)意味着有11行1列
    x_train = x_train.reshape(-1, 1)
    print(x_train.shape)

    y_values = [i for i in range(11)]
    # 但是这里有意使用x_values错误地创建了y_train，应该使用y_values，此处存在逻辑错误
    y_train = np.array(y_values, dtype=np.float32)
    # 同样，将目标值重塑为(-1, 1)的形式
    y_train = y_train.reshape(-1, 1)
    print(y_train.shape)

    # 设置输入和输出的维度
    input_dim = 1
    output_dim = 1
    # 创建一个线性回归模型实例
    model = LinearRegressionModel(input_dim, output_dim)
    model.to(device)

    # 设置学习率，用于模型参数更新的步长
    learning_rate = 0.001
    # 使用随机梯度下降（SGD）作为优化器来更新模型参数
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 使用均方误差（MSE）作为损失函数，评估模型预测值和真实值之间的误差
    criterion = nn.MSELoss()

    # 设置训练的迭代次数（即遍历数据集的次数）
    epochs = 2000
    for epoch in range(epochs):
        epoch += 1

        inputs = torch.from_numpy(x_train).to(device)
        labels = torch.from_numpy(y_train).to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
        if epoch % 50 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))

    predicted = model(torch.from_numpy(x_train).to(device)).cpu().data.numpy()
    print(predicted)

if __name__ == "__main__":
    # 打印当前安装的PyTorch版本
    print(torch.__version__)
    # 打印当前环境是否支持CUDA（用于GPU加速）
    print(torch.cuda.is_available())

    # 调用运行函数开始训练过程
    run()