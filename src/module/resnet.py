from torch import nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    r""" 设置模型参数是否需要梯度
    Args:
        model (torch.nn.Module): 模型
        feature_extracting (bool): 是否提取特征
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    r""" 初始化resnet模型
    Args:
        num_classes (int): 分类数
        feature_extract (bool): 是否提取特征
        use_pretrained (bool): 是否使用预训练模型

    Returns:
        model_ft (torch.nn.Module), params_to_update (generator)
    """

    model_ft = None

    if use_pretrained:
        # 使用预训练的权重，这里选择最符合需求的权重版本
        # 可以选择 ResNet152_Weights.IMAGENET1K_V1 或 ResNet152_Weights.DEFAULT
        # ResNet152_Weights.DEFAULT 通常会指向最新的预训练权重
        model_ft = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    else:
        # 不使用预训练的权重
        model_ft = models.resnet152(weights=None)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes),
                                nn.LogSoftmax(dim=1))

    # 获取要训练的层
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)

    return model_ft, params_to_update


