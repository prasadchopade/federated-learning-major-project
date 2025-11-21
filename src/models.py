import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50_model(num_classes=3, pretrained=True):
    """
    Get ResNet50 model for pneumonia classification

    Args:
        num_classes: Number of output classes (3: Normal, Pneumonia, Other)
        pretrained: Use ImageNet pretrained weights

    Returns:
        model: ResNet50 model
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Modify final layer for 3 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_model_weights(model):
    """Copy parameters and buffers, including BatchNorm running statistics."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}

def set_model_weights(model, weights):
    """Restore a complete model state without replacing optimizer parameters."""
    model.load_state_dict(weights, strict=True)

def get_model_parameters(model):
    """Get flattened model parameters for averaging"""
    params = []
    for param in model.parameters():
        params.append(param.data.flatten())
    return torch.cat(params)

def set_model_parameters(model, flat_params):
    """Set model parameters from flattened vector"""
    with torch.no_grad():
        offset = 0
        for param in model.parameters():
            param_size = param.data.numel()
            param.data = flat_params[offset:offset+param_size].reshape(param.data.shape)
            offset += param_size
