import torch.nn as nn
import timm

def get_xception_model(num_classes=2, dropout=0.4, pretrained=True):
    """
    Returns an Xception model from timm, modified for binary classification.

    Args:
        num_classes (int): Number of output classes (default 2).
        dropout (float): Dropout probability before final layer.
        pretrained (bool): Load pretrained weights on ImageNet.

    Returns:
        model (nn.Module): Modified Xception model.
    """
    model = timm.create_model('xception', pretrained=pretrained)
    
    # Reset the classifier head to desired number of classes
    model.reset_classifier(num_classes=num_classes)

    # Replace with dropout + linear explicitly
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.get_classifier().in_features, num_classes)
    )

    return model
