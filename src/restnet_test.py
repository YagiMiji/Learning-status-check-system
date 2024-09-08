# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import datasets,transforms
# # import matplotlib.pyplot as plt
# import numpy as np

# # 定义超参数
# input_size = 28  #图像的总尺寸28*28
# num_classes = 10  #标签的种类数
# num_epochs = 3  #训练的总循环周期
# batch_size = 64  #一个撮（批次）的大小，64张图片

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 训练集
# train_dataset = datasets.MNIST(root='./data',
#                             train=True,
#                             transform=transforms.ToTensor(),
#                             download=True)

# # 测试集
# test_dataset = datasets.MNIST(root='./data',
#                            train=False,
#                            transform=transforms.ToTensor())

# # 构建batch数据
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(         # 输入大小 (1, 28, 28)
#             nn.Conv2d(
#                 in_channels=1,              # 灰度图
#                 out_channels=16,            # 要得到几多少个特征图
#                 kernel_size=5,              # 卷积核大小
#                 stride=1,                   # 步长
#                 padding=2,                  # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
#             ),                              # 输出的特征图为 (16, 28, 28)
#             nn.ReLU(),                      # relu层
#             nn.MaxPool2d(kernel_size=2),    # 进行池化操作（2x2 区域）, 输出结果为： (16, 14, 14)
#         )
#         self.conv2 = nn.Sequential(         # 下一个套餐的输入 (16, 14, 14)
#             nn.Conv2d(16, 32, 5, 1, 2),     # 输出 (32, 14, 14)
#             nn.ReLU(),                      # relu层
#             nn.MaxPool2d(2),                # 输出 (32, 7, 7)
#         )
#         self.out = nn.Linear(32 * 7 * 7, 10)   # 全连接层得到的结果

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)           # flatten操作，结果为：(batch_size, 32 * 7 * 7)
#         output = self.out(x)
#         return output


# def accuracy(predictions, labels):
#     pred = torch.max(predictions.data, 1)[1]
#     rights = pred.eq(labels.data.view_as(pred)).sum()
#     return rights, len(labels)


# # 实例化
# net = CNN()
# net.to(device)
# #损失函数
# criterion = nn.CrossEntropyLoss()
# #优化器
# optimizer = optim.Adam(net.parameters(), lr=0.001) #定义优化器，普通的随机梯度下降算法

# #开始训练循环
# for epoch in range(num_epochs):
#     #当前epoch的结果保存下来
#     train_rights = []

#     for batch_idx, (data, target) in enumerate(train_loader):  #针对容器中的每一个批进行循环
#         net.train()
#         output = net(data.to(device))
#         loss = criterion(output, target.to(device))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         right = accuracy(output,  target.to(device))
#         train_rights.append(right)


#         if batch_idx % 100 == 0:

#             net.eval()
#             val_rights = []

#             for (data, target) in test_loader:
#                 output = net(data.to(device))
#                 right = accuracy(output, target.to(device))
#                 val_rights.append(right)

#             #准确率计算
#             train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
#             val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

#             print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
#                 epoch, batch_idx * batch_size, len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.data,
#                 100. * train_r[0].cpu().numpy() / train_r[1],
#                 100. * val_r[0].cpu().numpy() / val_r[1]))


import os
import numpy as np
import torch
import torch.utils.data
from torch import nn
import torch.optim as optim
import torchvision
# pip install torchvision
from torchvision import transforms, models, datasets
# https://pytorch.org/docs/stable/torchvision/index.html
# import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

# data_dir = 'flower_data/'
# train_dir = data_dir + '/train'
# valid_dir = data_dir + '/valid'
#
# data_transforms = {
#     'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
#                                  transforms.CenterCrop(224),  # 从中心开始裁剪
#                                  transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
#                                  transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
#                                  transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
#                                  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
#                                  transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
#                                  ]),
#     'valid': transforms.Compose([transforms.Resize(256),
#                                  transforms.CenterCrop(224),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                  ]),
# }
#
# batch_size = 8
#
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
#                ['train', 'valid']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
# class_names = image_datasets['train'].classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model_name = 'resnet'  # 可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
# 是否用人家训练好的特征来做
feature_extract = True

# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                    nn.LogSoftmax(dim=1))
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU计算
model_ft = model_ft.to(device)

# 模型保存
filename = 'checkpoint.pth'

# 是否训练所有层
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

print(model_ft)

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 学习率每7个epoch衰减成原来的1/10
# 最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename=filename):
    since = time.time()
    best_acc = 0
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders,
                                                                                             criterion,
                                                                                             optimizer_ft,
                                                                                             num_epochs=20,
                                                                                             is_inception=(model_name=="inception"))

for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有的参数，学习率调小一点
optimizer = optim.Adam(params_to_update, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.NLLLoss()

# Load the checkpoint

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
#model_ft.class_to_idx = checkpoint['mapping']

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=10, is_inception=(model_name=="inception"))

