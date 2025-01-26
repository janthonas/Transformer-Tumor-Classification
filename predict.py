from network_architecture import transform_pipeline, transfer_model

from PIL import Image
import torch

state_dict = torch.load("models/best_tumor_model.pth")
transfer_model.load_state_dict(state_dict)

labels = ['Healthy', 'Tumor']

filename = 'mri-tumor-org/test/Tumor/glioma (127).jpg'

img = Image.open(filename)
img = transform_pipeline(img)
img = img.unsqueeze(0)

prediction = transfer_model(img)
prediction = prediction.argmax()
print(labels[prediction])