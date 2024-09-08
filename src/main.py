import module.resnet as cnn

if __name__ == '__main__':
    model_ft, params_to_update = cnn.initialize_model(102, True, use_pretrained=True)
