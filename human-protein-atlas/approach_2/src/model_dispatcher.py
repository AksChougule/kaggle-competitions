import models as models

# Create dictionary which can dispacth models
MODEL_DISPATCHER = {
    'resnet34': models.ResNet34,
    'resnet50': models.ResNet50,
    'resnet50_v2': models.ResNet50_v2,
    'resnet50_v3': models.ResNet50_v3
}