# Import Necessary Libraries
import torchvision
from torchvision import datasets, transforms, models

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader

from PIL import Image

from timm import create_model


# Parsing Images

## Building Training Dataset
train_data_path = 'mri-tumor-org/train/'

# There is a set of transforms applied to the images
# - .ToTensor() converts images to tensors
# - .Resize(64) scales all images to 64 x 64 (god this is useful)
# - The values are normalized around means and std that keeps values between 0 and 1
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)), # used for transformer models that use 224x224
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    torchvision.transforms.RandomHorizontalFlip(p=0.1),
    torchvision.transforms.RandomVerticalFlip(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform_pipeline)

## Building Validation Dataset
val_data_path = 'mri-tumor-org/val/'
val_data = torchvision.datasets.ImageFolder(root=val_data_path,
                                            transform=transform_pipeline)

## Building Test Dataset
test_data_path = 'mri-tumor-org/test/'
test_data = torchvision.datasets.ImageFolder(root=test_data_path,
                                            transform=transform_pipeline)

## Building Data Loaders
batch_size = 64

train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Creating the Network

## Importing the Model
transfer_model = create_model('vit_base_patch16_224', pretrained=True)

## Freezing the Model Parameters except for BatchNorm
for name, param in transfer_model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False
        
# Network architecture using the syntax of transformers
transfer_model.head = nn.Sequential(
    # Reduces the transfer model outputs to 500
    nn.Linear(transfer_model.head.in_features, 500),
    nn.ReLU(),
    nn.Dropout(),
    # Reduces the model from 500 to 2 for the classification
    nn.Linear(500, 2)
)

# Designing the Optimizer
found_lr = 0.01

## For Transformer Models
## Separate the main body parameters and the head parameters
main_body_params = [p for name, p in transfer_model.named_parameters() if 'head' not in name]
head_params = transfer_model.head.parameters()

## Define the optimizer
optimizer = torch.optim.AdamW([
    # Parameters for the main transformer body
    {'params': main_body_params, 'lr': found_lr / 3},
    # Parameters for the classification head
    {'params': head_params, 'lr': found_lr / 9},
])
        
