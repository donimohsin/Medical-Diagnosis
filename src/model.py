"""Model construction / loading and the evaluation image transform."""
import torch, torch.nn as nn
from torchvision import models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE = 224

def eval_transform(size=IMG_SIZE):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])

def train_transform(size=IMG_SIZE):
    # No horizontal flip: in chest X-rays it would swap the side of the heart.
    return transforms.Compose([transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                               transforms.RandomRotation(10),
                               transforms.ColorJitter(brightness=0.15, contrast=0.15),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def build_model(arch, num_classes, pretrained=True):
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    elif arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError("arch must be resnet50 or resnet18")
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    return m

def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["classes"]
