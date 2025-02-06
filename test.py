from network_architecture import transform_pipeline, transfer_model

from PIL import Image
import torch

import os

state_dict = torch.load("models/best_tumor_model.pth")
transfer_model.load_state_dict(state_dict)

labels = ['Healthy', 'Tumor']

healthy_folder_path = "mri-tumor-org/test/Healthy"
healthy_file_names = [os.path.join(healthy_folder_path, f) for f in os.listdir(healthy_folder_path) if os.path.isfile(os.path.join(healthy_folder_path, f))]

tumor_folder_path = "mri-tumor-org/test/Tumor"
tumor_file_names = [os.path.join(tumor_folder_path, f) for f in os.listdir(tumor_folder_path) if os.path.isfile(os.path.join(tumor_folder_path, f))]

# Testing on healthy data
healthy_predict_list = []
for healthy in healthy_file_names:
    print(healthy)
    img = Image.open(healthy)
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    
    prediction = transfer_model(img)
    prediction = prediction.argmax()
    healthy_predict_list.append(labels[prediction])
    
# Testing on tumor data
tumor_predict_list = []
for tumor in tumor_file_names:
    print(tumor)
    img = Image.open(tumor)
    img = transform_pipeline(img)
    img = img.unsqueeze(0)
    
    prediction = transfer_model(img)
    prediction = prediction.argmax()
    tumor_predict_list.append(labels[prediction])
    
print(healthy_predict_list)
print(len(healthy_predict_list))

print(tumor_predict_list)
print(len(tumor_predict_list))
